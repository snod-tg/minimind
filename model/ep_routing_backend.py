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
    for local_id,count in enumerate(counts.tolist() if isinstance(counts,torch.Tensor) else counts):
        if count:
            values.append(owner.experts[str(owner.first_expert_id+local_id)](x[offset:offset+count]))
        offset+=count
    output_dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
    return torch.cat(values) if values else x.to(output_dtype)


class _ExpertReturn(torch.autograd.Function):
    """Attach one autograd node to prelaunched return bucket exchanges.

    A single backward node keeps collective order identical on every Rank,
    including when different experts are empty. Backward remains serialized.
    """
    @staticmethod
    def forward(ctx,group,input_splits,output_splits,stream,received,*sent):
        ctx.group=group;ctx.input_splits=input_splits;ctx.output_splits=output_splits;ctx.stream=stream
        return tuple(received)

    @staticmethod
    def backward(ctx,*grad_outputs):
        compute=torch.cuda.current_stream(grad_outputs[0].device);grads=[]
        ctx.stream.wait_stream(compute)
        with torch.cuda.stream(ctx.stream):
            for grad,send,recv in zip(grad_outputs,ctx.input_splits,ctx.output_splits):
                grad.record_stream(ctx.stream)
                value=grad.new_empty((sum(send),*grad.shape[1:]))
                with stage('pipeline.backward_return'):
                    dist.all_to_all_single(value,grad.contiguous(),output_split_sizes=list(send),
                        input_split_sizes=list(recv),group=ctx.group)
                value.record_stream(compute);grads.append(value)
        compute.wait_stream(ctx.stream)
        return (None,)*5+tuple(grads)

class MiniMindEPRouting:
    def __init__(self,model,library,mode='compact',async_counts=True,pipelined_return=False,return_bucket_bytes=16*1024*1024):
        if mode not in ('compact','dedup_train'): raise ValueError('supported modes: compact, dedup_train')
        from my_cuda_kernels.ep_ops import load_library
        load_library(library)
        self.mode=mode;self.enabled=True;self.workspaces={};self.scratch_spaces={};self.calls=0;self.fallbacks=0;self.transport_rows=0;self.original_transport_rows=0
        if return_bucket_bytes<=0:raise ValueError("return bucket size must be positive")
        self.return_bucket_bytes=return_bucket_bytes
        self.async_counts=async_counts;self.pipelined_return=pipelined_return
        self.streams={};self.host_counts={};self.return_calls=0;self.max_return_bytes=0
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

    def comm_stream(self,x):
        key=(x.device,torch.cuda.current_stream(x.device).cuda_stream)
        if key not in self.streams:self.streams[key]=torch.cuda.Stream(device=x.device)
        return self.streams[key]

    def count_table(self,owner,ids,wire_counts):
        # All shapes are static; scatter avoids bincount's dynamic-size readback.
        per_expert=torch.zeros(owner.config.num_experts,dtype=torch.int64,device=ids.device)
        per_expert.scatter_add_(0,ids,torch.ones_like(ids))
        return torch.cat((wire_counts[:,None],per_expert.view(owner.ep_size,owner.experts_per_rank)),dim=1)

    def exchange_counts(self,owner,x,table):
        # Only the pipeline gathers global counts so bucket boundaries agree.
        shape=(owner.ep_size,*table.shape) if self.pipelined_return else (2,*table.shape)
        received=table.new_empty(shape) if self.pipelined_return else torch.empty_like(table)
        def exchange():
            if self.pipelined_return:
                dist.all_gather_into_tensor(received.flatten(0,1),table,group=owner.process_group)
                return received
            dist.all_to_all_single(received,table,group=owner.process_group)
            return torch.stack((table,received))
        rank=dist.get_rank(owner.process_group)
        if not self.async_counts:return exchange(),None,rank
        stream=self.comm_stream(x);stream.wait_stream(torch.cuda.current_stream(x.device))
        key=(x.device,torch.cuda.current_stream(x.device).cuda_stream,shape,self.pipelined_return)
        if key not in self.host_counts:
            with torch.inference_mode(False):
                self.host_counts[key]=torch.empty(shape,dtype=torch.int64,pin_memory=True)
        host=self.host_counts[key]
        with torch.cuda.stream(stream):
            table.record_stream(stream);received.record_stream(stream)
            host.copy_(exchange(),non_blocking=True)
            ready=torch.cuda.Event();ready.record(stream)
        return host,ready,rank

    def read_counts(self,transfer):
        table,ready,rank=transfer
        if ready is not None:ready.synchronize()
        rows=table.tolist()
        if self.pipelined_return:return rows[rank],[source[rank] for source in rows],rows
        return rows[0],rows[1],None

    def return_plan(self,global_tables,row_bytes):
        """Return each complete expert in capacity-bounded communication buckets."""
        capacity=self.return_bucket_bytes//row_bytes
        if capacity<1:raise ValueError('return bucket must fit one output row')
        ranks=len(global_tables);experts=len(global_tables[0][0])-1
        cursors=[[0]*ranks for _ in range(ranks)];plan=[]
        for expert in range(experts):
            remaining=[[global_tables[peer][r][1+expert]
                for peer in range(ranks)] for r in range(ranks)]
            # Empty experts need no collective; every rank uses the same table.
            while any(any(row) for row in remaining):
                send_room=[capacity]*ranks;recv_room=[capacity]*ranks
                bucket=[[0]*ranks for _ in range(ranks)]
                for source in range(ranks):
                    for peer in range(ranks):
                        count=min(remaining[source][peer],send_room[source],recv_room[peer])
                        bucket[source][peer]=count;remaining[source][peer]-=count
                        send_room[source]-=count;recv_room[peer]-=count
                plan.append((bucket,[row.copy() for row in cursors]))
                for source in range(ranks):
                    for peer in range(ranks):cursors[source][peer]+=bucket[source][peer]
        return plan or [([[0]*ranks for _ in range(ranks)],cursors)]

    def return_experts(self,owner,x,counts,sends,recvs,send_perm,global_tables):
        """Pack output buckets directly from complete expert GEMMs.

        Compute each expert once in original order. Return its output before
        computing the next expert; split only communication at bucket capacity.
        """
        stream=self.comm_stream(x);compute=torch.cuda.current_stream(x.device)
        dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
        row_bytes=x.shape[1]*torch.empty((),dtype=dtype).element_size()
        with stage('pipeline.plan'):
            plan=self.return_plan(global_tables,row_bytes)
        rank=dist.get_rank(owner.process_group);experts=owner.experts_per_rank
        offsets=[0];send_offsets=[0]
        for e,count in enumerate(counts):
            offsets.append(offsets[-1]+count)
            send_offsets.append(send_offsets[-1]+sum(row[1+e] for row in sends))
        values=[];sent=[];received=[];input_splits=[];output_splits=[];maps=[]
        for bucket,cursors in plan:
            send=tuple(bucket[rank]);recv=tuple(bucket[peer][rank] for peer in range(owner.ep_size))
            with stage('pipeline.bucket_schedule'):
                parts=[]
                for peer in range(owner.ep_size):
                    lo=cursors[rank][peer];hi=lo+send[peer];peer_offset=0
                    map_lo=cursors[peer][rank];map_hi=map_lo+recv[peer];map_offset=0
                    for e in range(experts):
                        length=recvs[peer][1+e]
                        start=max(lo,peer_offset);end=min(hi,peer_offset+length)
                        if start<end:
                            # Full GEMM shapes and BF16 weight-gradient reductions
                            # stay unchanged even when a bucket splits its output.
                            while len(values)<=e:
                                i=len(values)
                                with stage('pipeline.expert_compute'):
                                    part=x[offsets[i]:offsets[i+1]]
                                    values.append(owner.experts[str(owner.first_expert_id+i)](part) if counts[i] else part.to(dtype))
                            expert_start=sum(row[1+e] for row in recvs[:peer])+start-peer_offset
                            parts.append(values[e][expert_start:expert_start+end-start])
                        peer_offset+=length
                        length=sends[peer][1+e]
                        start=max(map_lo,map_offset);end=min(map_hi,map_offset+length)
                        if start<end:
                            map_start=send_offsets[e]+sum(row[1+e] for row in sends[:peer])+start-map_offset
                            maps.append(send_perm[map_start:map_start+end-start])
                        map_offset+=length
            with stage('pipeline.bucket_pack'):
                value=torch.cat(parts) if parts else x[:0].to(dtype)
            sent.append(value);input_splits.append(send);output_splits.append(recv)
            actual=max(sum(send),sum(recv))*row_bytes
            assert actual<=self.return_bucket_bytes
            self.max_return_bytes=max(self.max_return_bytes,actual);self.return_calls+=1
            stream.wait_stream(compute)
            with torch.cuda.stream(stream),stage('pipeline.return_async'):
                value.record_stream(stream)
                returned=value.new_empty((sum(recv),value.shape[1]))
                dist.all_to_all_single(returned,value.detach(),output_split_sizes=list(recv),
                    input_split_sizes=list(send),group=owner.process_group)
            returned.record_stream(compute);received.append(returned)
        with stage('pipeline.restore'):
            returned=_ExpertReturn.apply(owner.process_group,tuple(input_splits),tuple(output_splits),stream,tuple(received),*sent)
            wire_perm=torch.cat(maps) if maps else send_perm[:0]
            inverse=torch.empty_like(wire_perm)
            inverse.scatter_(0,wire_perm,torch.arange(len(wire_perm),device=x.device,dtype=torch.int64))
            compute.wait_stream(stream)
            self.last_return_buckets=len(sent)
            return torch.ops.my_cuda_ep.gather(torch.cat(returned),inverse)

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
        with stage('counts_exchange'):
            table=self.count_table(owner,flat_ids,send_counts)
            transfer=self.exchange_counts(owner,x,table)
        if self.pipelined_return:
            send_perm,_,_=ws.partition(flat_ids[perm]%owner.experts_per_rank,1,'return_map',owner.experts_per_rank)
        with stage('pack'):
            if owner.training:
                send,send_ids=torch.ops.my_cuda_ep.pack(x,flat_ids,perm,inverse,topk,owner.experts_per_rank)
            else:
                send=ws.get('send_x',(n,h),x.dtype,device)
                send_ids=ws.get('send_ids',(n,),torch.int64,device)
                torch.ops.my_cuda_ep.pack_out(x,flat_ids,perm,topk,owner.experts_per_rank,send,send_ids)
        with stage('host_splits'):
            sends,recvs,global_tables=self.read_counts(transfer)
            send_splits=tuple(row[0] for row in sends);recv_splits=tuple(row[0] for row in recvs);r=sum(recv_splits)
            local_counts=tuple(sum(row[1+e] for row in recvs) for e in range(owner.experts_per_rank))
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
        if self.pipelined_return:
            returned=self.return_experts(owner,sorted_x,local_counts,sends,recvs,send_perm,global_tables)
        else:
            with stage('experts'):sorted_y=native_experts(owner,sorted_x,local_counts)
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
        with stage('counts_exchange'):
            table=self.count_table(owner,flat_ids,unique_counts)
            transfer=self.exchange_counts(owner,x,table)
        if self.pipelined_return:
            send_perm,_,_=ws.partition(flat_ids[perm]%experts,1,'return_map',experts)
        with stage('pack'):
            send,metadata,_=torch.ops.my_cuda_ep.dedup_pack(x,flat_ids,perm,flags,prefix,experts,bits)
        with stage('host_splits'):
            sends,recvs,global_tables=self.read_counts(transfer)
            send_unique=tuple(v[0] for v in sends);recv_unique=tuple(v[0] for v in recvs)
            send_expanded=tuple(sum(v[1:]) for v in sends);recv_expanded=tuple(sum(v[1:]) for v in recvs)
            total_send=sum(send_unique);total_recv=sum(recv_unique);expanded=sum(recv_expanded)
            local_counts=tuple(sum(row[1+e] for row in recvs) for e in range(experts))
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
        if self.pipelined_return:
            returned=self.return_experts(owner,sorted_x,local_counts,sends,recvs,send_perm,global_tables)
        else:
            with stage('experts'):sorted_y=native_experts(owner,sorted_x,local_counts)
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
        self.workspaces.clear();self.scratch_spaces.clear();self.streams.clear();self.host_counts.clear()


def attach_ep_routing(model, *, library=None, mode='compact',async_counts=True,pipelined_return=False,return_bucket_bytes=16*1024*1024):
    """Attach optimized routing without changing native expert computation."""
    return MiniMindEPRouting(model, library, mode,async_counts,pipelined_return,return_bucket_bytes)
