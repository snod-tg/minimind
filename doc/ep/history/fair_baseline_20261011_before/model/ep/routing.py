"""MiniMind routing, NCCL orchestration and native expert execution."""
import torch
import torch.distributed as dist
from my_cuda_kernels.ep import Workspace, combine_inference

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
    def __init__(self,model,library,mode='compact',decode_capacity_tokens=32):
        if mode not in ('compact','fixed','dedup'): raise ValueError('supported modes: compact, fixed, dedup')
        from my_cuda_kernels.ep_ops import load_library
        load_library(library)
        self.mode=mode;self.fixed_decode=False;self.decode_capacity_tokens=decode_capacity_tokens;self.enabled=True;self.workspaces={};self.calls=0;self.fallbacks=0
        if decode_capacity_tokens<1:raise ValueError("Decode capacity must be positive")
        self.layers=[layer.mlp for layer in model.model.layers]
        for i,mlp in enumerate(self.layers):
            if not hasattr(mlp,'local_expert_ids'):raise ValueError('EP model required')
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
        key=(owner._ep_routing_layer,x.device,torch.cuda.current_stream(x.device).cuda_stream)
        if key not in self.workspaces:self.workspaces[key]=Workspace()
        return self.workspaces[key]
    @torch.compiler.disable(recursive=False)
    def __call__(self,owner,x,ids,weights,scores=None):
        if not torch.compiler.is_compiling():self.calls+=1
        if self.mode=='fixed' and self.fixed_decode and not owner.training:
            return self.fixed(owner,x,ids,weights)
        if self.mode=='dedup' and ids.shape[1]==2 and not owner.training:
            return self.dedup(owner,x,ids,weights)
        ws=Workspace(reuse=False) if owner.training else self.workspace(owner,x);topk=ids.shape[1];n=len(x)*topk;h=x.shape[1]
        device=x.device;flat_ids=ids.reshape(-1)
        perm,inverse,send_counts=ws.sort(flat_ids,owner.experts_per_rank,'send',owner.ep_size)
        send=ws.get('send_x',(n,h),x.dtype,device)
        send_ids=ws.get('send_ids',(n,),torch.int64,device)
        if owner.training:
            send,send_ids=torch.ops.my_cuda_ep.pack(x,flat_ids,perm,inverse,topk,owner.experts_per_rank)
        else:
            torch.ops.my_cuda_ep.pack_out(x,flat_ids,perm,topk,owner.experts_per_rank,send,send_ids)
        recv_counts=ws.get('recv_counts',(owner.ep_size,),torch.int64,device)
        dist.all_to_all_single(recv_counts,send_counts,group=owner.process_group)
        send_splits=send_counts.tolist();recv_splits=recv_counts.tolist();r=sum(recv_splits)
        recv=ws.get('recv_x',(r,h),x.dtype,device)
        recv_ids=ws.get('recv_ids',(r,),torch.int64,device)
        if owner.training:
            recv=owner.communicate(send,recv_splits,send_splits,owner.process_group)
        else:
            dist.all_to_all_single(recv,send,output_split_sizes=recv_splits,input_split_sizes=send_splits,group=owner.process_group)
        dist.all_to_all_single(recv_ids,send_ids,output_split_sizes=recv_splits,input_split_sizes=send_splits,group=owner.process_group)
        ep_perm,ep_inverse,expert_counts=ws.sort(recv_ids,1,'expert',owner.experts_per_rank)
        sorted_x=ws.get('expert_x',(r,h),x.dtype,device)
        if owner.training:
            sorted_x=torch.ops.my_cuda_ep.gather(recv,ep_perm)
        else:
            torch.ops.my_cuda_ep.gather_out(recv,ep_perm,sorted_x)
        sorted_y=native_experts(owner,sorted_x,expert_counts)
        recv_order=ws.get('recv_order_y',(r,h),sorted_y.dtype,device)
        if owner.training:
            recv_order=torch.ops.my_cuda_ep.gather(sorted_y,ep_inverse)
        else:
            torch.ops.my_cuda_ep.gather_out(sorted_y,ep_inverse,recv_order)
        returned=ws.get('returned_y',(n,h),sorted_y.dtype,device)
        if owner.training:
            returned=owner.communicate(recv_order,send_splits,recv_splits,owner.process_group)
        else:
            dist.all_to_all_single(returned,recv_order,output_split_sizes=send_splits,input_split_sizes=recv_splits,group=owner.process_group)
        # Return owned storage so later layers/steps cannot overwrite this output.
        out=torch.empty_like(x)
        if owner.training:
            out=torch.ops.my_cuda_ep.combine(returned,weights.reshape(-1),inverse,topk,(torch.float32,torch.float16,torch.bfloat16).index(x.dtype))
        else:
            combine_inference(returned,weights.reshape(-1),inverse,topk,out)
        if owner.training and owner.config.router_aux_loss_coef>0:
            load=torch.nn.functional.one_hot(ids,owner.config.num_experts).float().mean(dim=(0,1))
            dist.all_reduce(load,op=dist.ReduceOp.SUM,group=owner.process_group);load.div_(owner.ep_size)
            owner.aux_loss=(load*scores.mean(dim=0)).sum()*owner.config.num_experts*owner.config.router_aux_loss_coef
        else:
            owner.aux_loss=weights.new_zeros(())
        return out
    def dedup(self,owner,x,ids,weights):
        # Reference experiment: one input vector per token/destination rank.
        # Outputs remain per assignment, so BF16 expert weighting is unchanged.
        ws=self.workspace(owner,x);device=x.device;h=x.shape[1];epr=owner.experts_per_rank
        flat_ids=ids.reshape(-1);_,inverse,assignment_counts=ws.sort(flat_ids,epr,'send',owner.ep_size)
        ranks=ids//epr;local=ids%epr;bits=(epr+1).bit_length();mask=(1<<bits)-1
        token_lists=[torch.nonzero((ranks==r).any(dim=1),as_tuple=True)[0] for r in range(owner.ep_size)]
        tokens=torch.cat(token_lists)
        meta_parts=[]
        for r,t in enumerate(token_lists):
            first=torch.where(ranks[t,0]==r,local[t,0]+1,0)
            second=torch.where(ranks[t,1]==r,local[t,1]+1,0)
            # Cross-rank second-slot assignment becomes the first wire expert.
            meta_parts.append(torch.where(first>0,first+(second<<bits),second))
        send_meta=torch.cat(meta_parts);send=x[tokens]
        send_unique=[len(t) for t in token_lists]
        # Exchange unique and expanded counts in one small collective.
        counts=ws.get('dedup_send_counts',(owner.ep_size,2),torch.int64,device)
        counts[:,0]=torch.tensor(send_unique,device=device);counts[:,1]=assignment_counts
        recv_counts=ws.get('dedup_recv_counts',(owner.ep_size,2),torch.int64,device)
        dist.all_to_all_single(recv_counts,counts,group=owner.process_group)
        send_assign=assignment_counts.tolist();recv_table=recv_counts.tolist()
        recv_unique=[v[0] for v in recv_table];recv_assign=[v[1] for v in recv_table]
        u=sum(recv_unique);r=sum(recv_assign)
        recv=ws.get('dedup_recv_x',(u,h),x.dtype,device);recv_meta=ws.get('dedup_recv_meta',(u,),torch.int64,device)
        dist.all_to_all_single(recv,send,output_split_sizes=recv_unique,input_split_sizes=send_unique,group=owner.process_group)
        dist.all_to_all_single(recv_meta,send_meta,output_split_sizes=recv_unique,input_split_sizes=send_unique,group=owner.process_group)
        lengths=1+(recv_meta>>bits>0).long()
        starts=torch.cumsum(lengths,0)-lengths
        wire=torch.repeat_interleave(torch.arange(u,device=device),lengths,output_size=r)
        slot=torch.arange(r,device=device)-starts[wire]
        expanded_ids=torch.where(slot==0,recv_meta[wire]&mask,recv_meta[wire]>>bits)-1
        ep_perm,ep_inverse,expert_counts=ws.sort(expanded_ids,1,'expert',epr)
        sorted_x=ws.get('dedup_expert_x',(r,h),x.dtype,device)
        torch.ops.my_cuda_ep.gather_out(recv,wire[ep_perm],sorted_x)
        values=[];offset=0
        for local_id,count in enumerate(expert_counts.tolist()):
            if count:values.append(owner.experts[str(owner.first_expert_id+local_id)](sorted_x[offset:offset+count]))
            offset+=count
        output_dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
        sorted_y=torch.cat(values) if values else torch.empty_like(sorted_x,dtype=output_dtype)
        recv_order=ws.get('dedup_recv_order',(r,h),sorted_y.dtype,device)
        torch.ops.my_cuda_ep.gather_out(sorted_y,ep_inverse,recv_order)
        returned=ws.get('dedup_returned',(len(x)*2,h),sorted_y.dtype,device)
        dist.all_to_all_single(returned,recv_order,output_split_sizes=send_assign,input_split_sizes=recv_assign,group=owner.process_group)
        out=torch.empty_like(x);torch.ops.my_cuda_ep.combine_out(returned,weights.reshape(-1),inverse,2,out)
        owner.aux_loss=weights.new_zeros(())
        self.last_transport=dict(assignments=sum(send_assign),unique=sum(send_unique),input_bytes=sum(send_unique)*h*x.element_size(),original_input_bytes=sum(send_assign)*h*x.element_size())
        return out
    def fixed(self,owner,x,ids,weights):
        # Set fixed_decode on all ranks for a synchronized Decode phase.
        # Capacity is static, not selected independently from a rank's route.
        topk=ids.shape[1];capacity=self.decode_capacity_tokens*topk
        if len(x)>self.decode_capacity_tokens:
            raise ValueError('Decode input exceeds configured fixed capacity')
        ws=self.workspace(owner,x);n=len(x)*topk;h=x.shape[1];device=x.device
        flat_ids=ids.reshape(-1);perm,_,counts=ws.sort(flat_ids,owner.experts_per_rank,'send',owner.ep_size)
        send=ws.get('fixed_send',(owner.ep_size*capacity,h),x.dtype,device)
        send_ids=ws.get('fixed_send_ids',(owner.ep_size*capacity,),torch.int64,device)
        inverse=ws.get('fixed_inverse',(n,),torch.int64,device)
        torch.ops.my_cuda_ep.pack_fixed_out(x,flat_ids,perm,counts,topk,owner.experts_per_rank,send,send_ids,inverse)
        recv=ws.get('fixed_recv',(owner.ep_size*capacity,h),x.dtype,device)
        recv_ids=ws.get('fixed_recv_ids',(owner.ep_size*capacity,),torch.int64,device)
        dist.all_to_all_single(recv,send,group=owner.process_group)
        dist.all_to_all_single(recv_ids,send_ids,group=owner.process_group)
        ep_perm,ep_inverse,expert_counts=ws.sort(recv_ids,1,'fixed_expert',owner.experts_per_rank,masked=True)
        # The only host count read: preserve original compact PyTorch GEMM shape.
        counts_host=expert_counts.tolist();r=sum(counts_host)
        sorted_x=ws.get('fixed_expert_x',(r,h),x.dtype,device)
        torch.ops.my_cuda_ep.gather_out(recv,ep_perm[:r],sorted_x)
        values=[];offset=0
        for local_id,count in enumerate(counts_host):
            if count:values.append(owner.experts[str(owner.first_expert_id+local_id)](sorted_x[offset:offset+count]))
            offset+=count
        output_dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
        sorted_y=torch.cat(values) if values else torch.empty_like(sorted_x,dtype=output_dtype)
        recv_order=ws.get('fixed_recv_order',(owner.ep_size*capacity,h),sorted_y.dtype,device)
        torch.ops.my_cuda_ep.scatter_fixed_out(sorted_y,ep_inverse,recv_order)
        returned=ws.get('fixed_returned',(owner.ep_size*capacity,h),sorted_y.dtype,device)
        dist.all_to_all_single(returned,recv_order,group=owner.process_group)
        out=torch.empty_like(x)
        combine_inference(returned,weights.reshape(-1),inverse,topk,out)
        owner.aux_loss=weights.new_zeros(())
        return out
    def uninstall(self):
        for mlp in self.layers:
            del mlp._ep_routing_backend;del mlp._ep_routing_layer
        self.workspaces.clear()
