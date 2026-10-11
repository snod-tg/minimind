"""Optional forward-only backend for the verified MiniMind EP source snapshot.

Routing/dispatch/combine statements are reused from the original forward AST.
Training and calls outside inference_mode always execute the original method.
"""
import ast,contextlib,copy,inspect,textwrap,types
import torch
from .kernel9_mlp import GroupedExpertMLP


def compile_forward(original,audited):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn=tree.body[0];fn.decorator_list=[]
    boundaries=[(0,'router'),(29,'rank_pack'),(40,'count_exchange'),(46,'split_to_cpu'),
                (49,'dispatch'),(70,'expert_sort'),(75,'expert_counts'),(80,'experts'),
                (97,'inverse_expert'),(104,'combine'),(112,'restore')]
    groups={label:[] for _,label in boundaries}
    for node in fn.body:
        offset=node.lineno-1
        label=next(label for start,label in reversed(boundaries) if offset>=start)
        groups[label].append(node)
    count=groups['expert_counts'][0]
    assert isinstance(count,ast.Assign) and count.targets[0].id=='local_expert_counts'
    assert isinstance(count.value,ast.Call) and count.value.func.attr=='tolist'
    assert groups['experts'][0].targets[0].id=='expert_outputs'
    assert groups['combine'][0].targets[0].id=='returned_sorted_output'
    count.value=count.value.func.value  # retain the GPU bincount tensor
    groups['experts']=ast.parse('expert_sorted_output = self._ep_backend.experts(self, expert_sorted_x, local_expert_counts)').body
    if audited:
        groups['router']+=ast.parse('self._ep_backend.router(self, topk_expert_id, topk_weight)').body
        groups['experts']+=ast.parse('self._ep_backend.collect(self, expert_sorted_x, local_expert_counts, expert_sorted_output)').body
    fn.body=[]
    for _,label in boundaries:
        if audited:
            region=ast.parse(f"self._ep_backend.region('{label}', self._ep_layer)",mode='eval').body
            fn.body.append(ast.With(items=[ast.withitem(region,None)],body=groups[label]))
        else:fn.body+=groups[label]
    ast.fix_missing_locations(tree);namespace=dict(original.__globals__)
    exec(compile(tree,'<minimind_ep_backend>','exec'),namespace)
    return namespace['forward']


class MiniMindEPBackend:
    """Single invocation/stream per model; invalidate explicitly after weight updates."""
    def __init__(self,model,backend='original',threshold=256):
        if backend not in ('original','kernel9','hybrid','last_layer'):raise ValueError('unknown backend')
        self.model=model;self.backend=backend;self.threshold=threshold
        self.states={};self.capture=False;self.profile=False;self.rows={};self.routes={};self.calls={};self.prepare_events=0
        self.layers=[layer.mlp for layer in model.model.layers]
        for i,mlp in enumerate(self.layers):
            if hasattr(mlp,'_ep_backend'):raise RuntimeError('backend already installed')
            mlp._ep_original=mlp.forward;mlp._ep_layer=i;mlp._ep_backend=self
            function=mlp.forward.__func__
            mlp._ep_fast=types.MethodType(compile_forward(function,False),mlp)
            mlp._ep_audit=types.MethodType(compile_forward(function,True),mlp)
            mlp.forward=types.MethodType(self._forward,mlp)

    @staticmethod
    def _forward(mlp,x):
        controller=mlp._ep_backend
        if mlp.training or not torch.is_inference_mode_enabled():
            if mlp.training:controller.invalidate(mlp._ep_layer,'training call')
            return mlp._ep_original(x)
        if controller.capture or controller.profile:return mlp._ep_audit(x)
        if controller.backend=='original':return mlp._ep_original(x)
        if controller.backend=='last_layer' and mlp._ep_layer!=len(controller.layers)-1:return mlp._ep_original(x)
        return mlp._ep_fast(x)

    def initialize(self):
        for mlp in self.layers:self.state(mlp)

    def state(self,mlp):
        i=mlp._ep_layer
        if i not in self.states:
            experts={j:mlp.experts[str(j)] for j in mlp.local_expert_ids}
            device=next(mlp.experts.parameters()).device
            self.states[i]=GroupedExpertMLP(experts,mlp.local_expert_ids,device=device,preallocate=True)
        return self.states[i]

    def invalidate(self,layer=None,reason='weights changed'):
        for i in list(self.states):
            if layer is None or layer==i:self.states.pop(i).invalidate(reason)

    def release_workspaces(self):
        for s in self.states.values():
            s.shape=None;s.workspaces=None;s.activated=None;s.hidden=None

    def experts(self,mlp,x,counts):
        use=self.backend!='original' and x.dtype==torch.bfloat16
        if self.backend in ('hybrid','last_layer'):use=use and len(x)<=self.threshold
        if self.backend=='last_layer':use=use and mlp._ep_layer==len(self.layers)-1
        if use:
            state=self.state(mlp)
            shape=(tuple(x.shape),x.device,x.dtype)
            if state.shape!=shape:
                state.prepare(x);self.prepare_events+=1  # dynamic-M preparation stays inside forward
            y=state(x,counts);path='kernel9'
        else:
            values=[];offset=0
            for expert_id,m in zip(mlp.local_expert_ids,counts.tolist()):
                if m:values.append(mlp.experts[str(expert_id)](x[offset:offset+m]))
                offset+=m
            y=torch.cat(values) if values else x.clone();path='original'
        if self.capture:self.calls[mlp._ep_layer]=path
        return y

    def reset_capture(self):self.rows={};self.routes={};self.calls={}

    def router(self,mlp,ids,weights):
        if self.capture:self.routes[mlp._ep_layer]=dict(ids=ids.detach().cpu(),weights=weights.detach().cpu())

    def collect(self,mlp,x,counts,y):
        if self.capture:self.rows[mlp._ep_layer]=dict(m=len(x),counts=counts.cpu().tolist(),dtype=str(x.dtype),
                                                   input=x.detach().cpu(),output=y.detach().cpu(),path=self.calls[mlp._ep_layer])

    def region(self,name,layer):
        return torch.profiler.record_function(f'ep/{self.backend}/L{layer}/{name}') if self.profile else contextlib.nullcontext()
