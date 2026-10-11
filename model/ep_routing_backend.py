"""MiniMind routing, NCCL orchestration and native expert execution."""
import os
from dataclasses import dataclass
from contextlib import nullcontext
import torch
import torch.distributed as dist
from model.model_minimind import moe_aux_loss
from my_cuda_kernels.ep import Workspace, combine_inference

_PROFILE=os.environ.get('MINIMIND_EP_PROFILE')=='1'
def stage(name):
    return torch.profiler.record_function('ep.'+name) if _PROFILE else nullcontext()

@dataclass
class RoutingHandle:
    assignment_inverse: torch.Tensor
    expert_inverse: torch.Tensor
    send_splits: tuple
    recv_splits: tuple

@torch.compiler.disable(recursive=False)
def native_experts(owner,x,counts):
    # Host-dependent empty-expert branch stays outside capture. Each expert's
    # tensor-only forward/backward can still compile with dynamic row count.
    values=[];offset=0
    for local_id,count in enumerate(counts.tolist()):
        if count:
            values.append(owner.experts[str(owner.first_expert_id+local_id)](x[offset:offset+count]))
        offset+=count
    output_dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
    return torch.cat(values) if values else x.to(output_dtype)

class MiniMindEPRouting:
    def __init__(self,model,library,mode='compact'):
        if mode not in ('compact','dedup_train'): raise ValueError('supported modes: compact, dedup_train')
        from my_cuda_kernels.ep_ops import load_library
        load_library(library)
        self.mode=mode;self.enabled=True;self.workspaces={};self.scratch_spaces={};self.calls=0;self.fallbacks=0;self.transport_rows=0;self.original_transport_rows=0
        self.layers=[layer.mlp for layer in model.model.layers]
        for i,mlp in enumerate(self.layers):
            if not hasattr(mlp,'local_expert_ids'):raise ValueError('EP model required')
            if mode=='dedup_train' and mlp.config.num_experts_per_tok!=2:raise ValueError('dedup_train requires Top-2')
            experts=mlp.config.num_experts;topk=mlp.config.num_experts_per_tok;ranks=mlp.ep_size
            if experts%ranks or not 1<=topk<=experts:
                raise ValueError('expert count must be divisible by EP size and 1 <= Top-K <= experts')
            if not 1<=ranks<=1024 or not 1<=mlp.experts_per_rank<=1024:
                raise ValueError('native partition supports 1..1024 ranks/local experts')
            if getattr(mlp,'_ep_routing_backend',None) is not None:raise RuntimeError('EP routing already attached')
            object.__setattr__(mlp,'_ep_routing_backend',self)
            mlp._ep_routing_layer=i
    def can_use(self,owner,x):
        ok=(self.enabled and (owner.training or torch.is_inference_mode_enabled())
            and x.is_cuda and x.dtype in (torch.float32,torch.float16,torch.bfloat16)
            and 1<=owner.ep_size<=1024 and 1<=owner.experts_per_rank<=1024
            and owner.config.num_experts==owner.ep_size*owner.experts_per_rank
            and 1<=owner.config.num_experts_per_tok<=owner.config.num_experts and owner.config.ep_comm_dtype=='native')
        if self.enabled and owner.training and not ok:
            raise RuntimeError('optimized EP training requires CUDA, valid expert/Top-K/Rank configuration and native communication')
        if not ok and not torch.compiler.is_compiling():self.fallbacks+=1
        return ok
    def workspace(self,owner,x):
        key=(x.device,torch.cuda.current_stream(x.device).cuda_stream)
        if key not in self.workspaces:self.workspaces[key]=Workspace()
        return self.workspaces[key]
    def scratch(self,x):
        key=(x.device,torch.cuda.current_stream(x.device).cuda_stream)
        if key not in self.scratch_spaces:self.scratch_spaces[key]=Workspace()
        return self.scratch_spaces[key]

    @torch.compiler.disable(recursive=False)
    def __call__(self,owner,x,ids,weights,scores=None):
        if not torch.compiler.is_compiling():self.calls+=1
        if self.mode=='dedup_train':return self.dedup_training(owner,x,ids,weights,scores)
        ws=self.scratch(x) if owner.training else self.workspace(owner,x)
        topk=ids.shape[1];n=len(x)*topk;h=x.shape[1];device=x.device
        flat_ids=ids.reshape(-1)
        with stage('rank_layout'):
            if owner.training:perm,inverse,send_counts=ws.partition(flat_ids,owner.experts_per_rank,'send',owner.ep_size)
            else:perm,inverse,send_counts=ws.sort(flat_ids,owner.experts_per_rank,'send',owner.ep_size)
        with stage('pack'):
            if owner.training:
                send,send_ids=torch.ops.my_cuda_ep.pack(x,flat_ids,perm,inverse,topk,owner.experts_per_rank)
            else:
                send=ws.get('send_x',(n,h),x.dtype,device)
                send_ids=ws.get('send_ids',(n,),torch.int64,device)
                torch.ops.my_cuda_ep.pack_out(x,flat_ids,perm,topk,owner.experts_per_rank,send,send_ids)
        with stage('counts_exchange'):
            recv_counts=ws.get('recv_counts',(owner.ep_size,),torch.int64,device)
            dist.all_to_all_single(recv_counts,send_counts,group=owner.process_group)
        with stage('host_splits'):
            send_splits=tuple(send_counts.tolist());recv_splits=tuple(recv_counts.tolist());r=sum(recv_splits)
        with stage('dispatch'):
            if owner.training:recv=owner.communicate(send,recv_splits,send_splits,owner.process_group)
            else:
                recv=ws.get('recv_x',(r,h),x.dtype,device)
                dist.all_to_all_single(recv,send,output_split_sizes=list(recv_splits),input_split_sizes=list(send_splits),group=owner.process_group)
            recv_ids=ws.get('recv_ids',(r,),torch.int64,device)
            dist.all_to_all_single(recv_ids,send_ids,output_split_sizes=list(recv_splits),input_split_sizes=list(send_splits),group=owner.process_group)
        with stage('expert_layout'):
            if owner.training:
                ep_perm,ep_inverse,counts=ws.partition(recv_ids,1,'expert',owner.experts_per_rank)
                sorted_x=torch.ops.my_cuda_ep.gather(recv,ep_perm)
            else:
                ep_perm,ep_inverse,counts=ws.sort(recv_ids,1,'expert',owner.experts_per_rank)
                sorted_x=ws.get('expert_x',(r,h),x.dtype,device)
                torch.ops.my_cuda_ep.gather_out(recv,ep_perm,sorted_x)
        handle=RoutingHandle(inverse,ep_inverse,send_splits,recv_splits)
        with stage('experts'):sorted_y=native_experts(owner,sorted_x,counts)
        with stage('return'):
            if owner.training:
                recv_order=torch.ops.my_cuda_ep.gather(sorted_y,handle.expert_inverse)
                returned=owner.communicate(recv_order,handle.send_splits,handle.recv_splits,owner.process_group)
            else:
                recv_order=ws.get('recv_order_y',(r,h),sorted_y.dtype,device)
                torch.ops.my_cuda_ep.gather_out(sorted_y,handle.expert_inverse,recv_order)
                returned=ws.get('returned_y',(n,h),sorted_y.dtype,device)
                dist.all_to_all_single(returned,recv_order,output_split_sizes=list(handle.send_splits),input_split_sizes=list(handle.recv_splits),group=owner.process_group)
        with stage('combine'):
            if owner.training:
                out=torch.ops.my_cuda_ep.combine(returned,weights.reshape(-1),handle.assignment_inverse,topk,(torch.float32,torch.float16,torch.bfloat16).index(x.dtype))
            else:
                out=torch.empty_like(x)
                combine_inference(returned,weights.reshape(-1),handle.assignment_inverse,topk,out)
        owner.aux_loss=moe_aux_loss(scores,ids,owner.config,owner.process_group) if owner.training and owner.config.router_aux_loss_coef>0 else weights.new_zeros(())
        return out

    @torch.compiler.disable(recursive=False)
    def dedup_training(self,owner,x,ids,weights,scores=None):
        """Transmit once per token/rank; expand directly to expert input.

        Expanded expert rows and return order match compact exactly. The
        forward handle owns maps; no routing decision is reused across calls.
        """
        ws=self.scratch(x);device=x.device;n=len(x)*2;h=x.shape[1]
        experts=owner.experts_per_rank;bits=(experts+1).bit_length()
        flat_ids=ids.reshape(-1)
        with stage('rank_layout'):
            perm,inverse,assignment_counts=ws.partition(flat_ids,experts,'send',owner.ep_size)
            flags=ws.get('unique_flags',(n,),torch.int32,device)
            unique_counts=ws.get('unique_counts',(owner.ep_size,),torch.int64,device)
            torch.ops.my_cuda_ep.dedup_flags_out(flat_ids,perm,experts,flags,unique_counts)
            prefix=torch.cumsum(flags,dim=0,dtype=torch.int64)
        with stage('pack'):
            send,metadata,_=torch.ops.my_cuda_ep.dedup_pack(x,flat_ids,perm,flags,prefix,experts,bits)
        with stage('counts_exchange'):
            table=torch.stack((unique_counts,assignment_counts),dim=1)
            received=torch.empty_like(table)
            dist.all_to_all_single(received,table,group=owner.process_group)
        with stage('host_splits'):
            sends=table.tolist();recvs=received.tolist()
            send_unique=tuple(v[0] for v in sends);recv_unique=tuple(v[0] for v in recvs)
            send_expanded=tuple(v[1] for v in sends);recv_expanded=tuple(v[1] for v in recvs)
            total_send=sum(send_unique);total_recv=sum(recv_unique);expanded=sum(recv_expanded)
        with stage('dispatch'):
            recv=owner.communicate(send[:total_send],recv_unique,send_unique,owner.process_group)
            recv_meta=torch.empty((total_recv,),dtype=torch.int64,device=device)
            dist.all_to_all_single(recv_meta,metadata[:total_send],output_split_sizes=list(recv_unique),input_split_sizes=list(send_unique),group=owner.process_group)
        with stage('expert_layout'):
            ends=torch.cumsum(1+(recv_meta>>bits>0).long(),dim=0)
            expert_ids=torch.empty((expanded,),dtype=torch.int64,device=device)
            wire=torch.empty_like(expert_ids)
            torch.ops.my_cuda_ep.expand_layout_out(recv_meta,ends,bits,expert_ids,wire)
            ep_perm,ep_inverse,counts=ws.partition(expert_ids,1,'expert',experts)
            # No expanded payload intermediary: one gather reads unique recv.
            sorted_x=torch.ops.my_cuda_ep.expand(recv,wire[ep_perm])
        handle=RoutingHandle(inverse,ep_inverse,send_expanded,recv_expanded)
        with stage('experts'):sorted_y=native_experts(owner,sorted_x,counts)
        with stage('return'):
            recv_order=torch.ops.my_cuda_ep.gather(sorted_y,handle.expert_inverse)
            returned=owner.communicate(recv_order,handle.send_splits,handle.recv_splits,owner.process_group)
        with stage('combine'):
            out=torch.ops.my_cuda_ep.combine(returned,weights.reshape(-1),handle.assignment_inverse,2,(torch.float32,torch.float16,torch.bfloat16).index(x.dtype))
        owner.aux_loss=moe_aux_loss(scores,ids,owner.config,owner.process_group) if owner.training and owner.config.router_aux_loss_coef>0 else weights.new_zeros(())
        self.last_transport=dict(assignments=n,unique=total_send,input_bytes=total_send*h*x.element_size(),original_input_bytes=n*h*x.element_size())
        self.transport_rows+=total_send;self.original_transport_rows+=n
        return out

    def uninstall(self):
        for mlp in self.layers:
            del mlp._ep_routing_backend;del mlp._ep_routing_layer
        self.workspaces.clear();self.scratch_spaces.clear()


def attach_ep_routing(model, *, library=None, mode='compact'):
    """Attach optimized routing without changing native expert computation."""
    return MiniMindEPRouting(model, library, mode)
