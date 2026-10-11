"""Canonical KN linear checkpoints, with native NK inference compatibility.

Default forward preserves the original contiguous-NK PyTorch operation.
Canonical weights are directly available to KN CUDA kernels. Embeddings remain
native unless tied to a linear; tied wrappers preserve one shared parameter.
Optimizer/resume state and quantized/LoRA/HF export formats are not converted.
"""
from pathlib import Path
import re
import torch.distributed as dist
import torch
from torch import nn
from torch.nn import functional as F
FORMAT="minimind-linear-kn-v1"

class _KNWeight(nn.Module):
    def _setup(self,weight):
        self.weight=weight
        object.__setattr__(self,'_native_weight',None)
        self._stamp=None
        self.register_load_state_dict_post_hook(lambda module,incompatible:module.invalidate())

    def invalidate(self):
        object.__setattr__(self,'_native_weight',None)
        self._stamp=None
        torch.clear_autocast_cache()

    def _nk(self):
        if torch.is_grad_enabled():
            return self.weight.T.contiguous()
        w=self.weight
        stamp=(id(w),w.data_ptr(),w._version,w.device,w.dtype)
        if stamp!=self._stamp:
            if w.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('prepare KN native caches before graph capture')
            if self._stamp is not None:torch.clear_autocast_cache()
            with torch.inference_mode(False),torch.no_grad():
                cached=nn.Parameter(w.T.contiguous(),requires_grad=w.requires_grad)
            object.__setattr__(self,'_native_weight',cached)
            self._stamp=stamp
        return self._native_weight

class LinearKN(_KNWeight):
    def __init__(self,original,weight):
        super().__init__();self.in_features=original.in_features;self.out_features=original.out_features
        self._setup(weight);self.bias=original.bias;self.train(original.training)
    def forward(self,x):
        return F.linear(x,self._nk(),self.bias)

class TiedEmbeddingKN(_KNWeight):
    def __init__(self,original,weight):
        super().__init__()
        for name in ('num_embeddings','embedding_dim','padding_idx','max_norm','norm_type','scale_grad_by_freq','sparse'):
            setattr(self,name,getattr(original,name))
        if self.max_norm is not None:
            raise ValueError('tied embedding max_norm would mutate a detached inference cache')
        self._setup(weight);self.train(original.training)
    def forward(self,x):
        return F.embedding(x,self._nk(),self.padding_idx,self.max_norm,self.norm_type,self.scale_grad_by_freq,self.sparse)

def layout_spec(model):
    """Discover semantic module types, including shared embedding aliases."""
    if any(isinstance(m,_KNWeight) for m in model.modules()):
        raise ValueError('layout_spec requires an original NK model')
    params=dict(model.named_parameters(remove_duplicate=False))
    linear={f'{name}.weight' if name else 'weight' for name,m in model.named_modules() if isinstance(m,nn.Linear)}
    ids={id(params[key]) for key in linear}
    keys=sorted(key for key,p in params.items() if id(p) in ids)
    groups={}
    for key,p in params.items():groups.setdefault(id(p),[]).append(key)
    aliases=[v for v in groups.values() if len(v)>1]
    return dict(kn_keys=keys,linear_keys=sorted(linear),aliases=aliases)

def encode_state(state,spec):
    if set(spec['kn_keys'])-set(state):raise ValueError('missing canonical weight keys')
    for group in spec['aliases']:
        if any(not torch.equal(state[group[0]],state[key]) for key in group[1:]):
            raise ValueError('tied checkpoint aliases have different values')
    kn=set(spec['kn_keys']);encoded={}
    for key,value in state.items():
        if not isinstance(value,torch.Tensor):raise ValueError('model state must contain tensors')
        encoded[key]=(value.T if key in kn else value).detach().cpu().contiguous()
    for group in spec['aliases']:
        for key in group[1:]:encoded[key]=encoded[group[0]]
    return dict(format=FORMAT,layout=spec,state_dict=encoded)

def decode_state(payload):
    _check(payload);kn=set(payload['layout']['kn_keys'])
    return {key:(value.T.contiguous() if key in kn else value) for key,value in payload['state_dict'].items()}

def _check(payload):
    if not isinstance(payload,dict) or payload.get('format')!=FORMAT:
        raise ValueError('expected explicitly marked KN checkpoint')
    if not isinstance(payload.get('state_dict'),dict):raise ValueError('missing state_dict')
    keys=payload['layout']['kn_keys']
    if set(keys)-set(payload['state_dict']):raise ValueError('missing KN weights')
    for group in payload['layout']['aliases']:
        if any(not torch.equal(payload['state_dict'][group[0]],payload['state_dict'][key]) for key in group[1:]):
            raise ValueError('KN checkpoint has inconsistent tied aliases')

def convert_file(source,target,original_model):
    if Path(target).exists():raise FileExistsError(target)
    state=torch.load(source,map_location='cpu',weights_only=True)
    if isinstance(state,dict) and state.get('format')==FORMAT:raise ValueError('already KN; refusing double transpose')
    spec=layout_spec(original_model)
    expected=original_model.state_dict()
    if state.keys()!=expected.keys() or any(state[k].shape!=expected[k].shape for k in expected):
        raise ValueError('checkpoint differs from supplied original model')
    payload=encode_state(state,spec)
    if hasattr(original_model,'config'):payload['config']=original_model.config.to_dict()
    torch.save(payload,target)
    return payload

def canonicalize_model(model,spec=None):
    if getattr(model,'_kn_layout',None) is not None:
        existing=model._kn_layout
        if spec is not None and (set(spec['kn_keys'])!=set(existing['kn_keys']) or spec['aliases']!=existing['aliases']):
            raise ValueError('canonical model layout differs from checkpoint')
        return model
    discovered=layout_spec(model)
    if spec is not None and (set(spec['kn_keys'])!=set(discovered['kn_keys']) or spec['aliases']!=discovered['aliases']):
        raise ValueError('model architecture/ties do not match KN layout')
    spec=discovered;params=dict(model.named_parameters(remove_duplicate=False));ids={id(params[k]) for k in spec['kn_keys']}
    canonical={}
    with torch.inference_mode(False),torch.no_grad():
        for key in spec['kn_keys']:
            old=params[key]
            if id(old) not in canonical:canonical[id(old)]=nn.Parameter(old.T.contiguous(),requires_grad=old.requires_grad)
    for name,old in list(model.named_modules()):
        if not name:continue
        if isinstance(old,nn.Linear):new=LinearKN(old,canonical[id(old.weight)])
        elif isinstance(old,nn.Embedding) and id(old.weight) in ids:new=TiedEmbeddingKN(old,canonical[id(old.weight)])
        else:continue
        parent,_,child=name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model,child,new)
    model._kn_layout=spec
    return model

def load_kn_checkpoint(model,path,ep_subset=False):
    payload=torch.load(path,map_location='cpu',weights_only=True);_check(payload)
    spec=payload['layout'];state=payload['state_dict']
    if ep_subset:
        if not getattr(model.config,'use_ep',False):raise ValueError('ep_subset requires an EP model')
        expected=set(model.state_dict());local=getattr(model,'_kn_layout',None) or layout_spec(model)
        if set(local['kn_keys'])!=set(spec['kn_keys'])&expected:raise ValueError('EP KN layout mismatch')
        if local['aliases']!=spec['aliases']:raise ValueError('EP tied aliases mismatch')
        state={key:state[key] for key in expected};spec=local
    canonicalize_model(model,spec)
    model.load_state_dict(state,strict=True)
    return model

def _full_ep_spec(spec,num_experts):
    expanded=dict(spec)
    for field in ('kn_keys','linear_keys'):
        keys=set(spec[field])
        for key in spec[field]:
            match=re.search(r'(\.experts\.)[0-9]+(\.)',key)
            if match:
                for e in range(num_experts):
                    keys.add(key[:match.start()]+match.group(1)+str(e)+match.group(2)+key[match.end():])
        expanded[field]=sorted(keys)
    return expanded

def _save_payload(model,state,spec,path,canonical):
    if canonical:
        payload=dict(format=FORMAT,layout=spec,state_dict=state)
        for group in spec['aliases']:
            for key in group[1:]:state[key]=state[group[0]]
        _check(payload)
    else:payload=encode_state(state,spec)
    if hasattr(model,'config'):payload['config']=model.config.to_dict()
    torch.save(payload,path)

def save_kn_checkpoint(model,path):
    """All linear parameters save KN. EP: all ranks call; rank 0 saves full model."""
    canonical=getattr(model,'_kn_layout',None) is not None
    spec=model._kn_layout if canonical else layout_spec(model)
    params=dict(model.named_parameters(remove_duplicate=False))
    for group in spec['aliases']:
        if any(params[key] is not params[group[0]] for key in group[1:]):
            raise ValueError('model has lost its tied parameter identity')
    state={key:value.detach().cpu().contiguous() for key,value in model.state_dict().items()}
    if getattr(getattr(model,'config',None),'use_ep',False):
        if not dist.is_initialized():raise RuntimeError('EP save requires all distributed ranks')
        rank=dist.get_rank();shards=[None]*dist.get_world_size() if rank==0 else None
        dist.gather_object(state,shards,dst=0)
        status=[None]
        if rank==0:
            try:
                full={}
                for shard in shards:
                    for key,value in shard.items():
                        if key in full and not torch.equal(full[key],value):
                            raise ValueError('EP replicated weights differ: '+key)
                        full[key]=value
                spec=_full_ep_spec(spec,model.config.num_experts)
                _save_payload(model,full,spec,path,canonical)
            except Exception as exc:status[0]=str(exc)
        dist.broadcast_object_list(status,src=0)
        if status[0] is not None:raise RuntimeError('EP KN save failed: '+status[0])
    else:_save_payload(model,state,spec,path,canonical)

def prepare_native_caches(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module,_KNWeight):module._nk()
    return model
