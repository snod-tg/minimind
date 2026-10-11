"""Load native EP ops and declare their out-tensor mutations to FakeTensor."""
import torch
_REGISTERED=False

def load_library(path):
    global _REGISTERED
    torch.ops.load_library(str(path))
    if _REGISTERED:return
    def fake_out(*args):return None
    for name in ('sort_out','sort_masked_out','pack_out','gather_out','combine_out','masked_sort_out','pack_fixed_out','scatter_fixed_out'):
        torch.library.register_fake('my_cuda_ep::'+name)(fake_out)
    _REGISTERED=True

# Functional training operators expose allocation/gradient contracts to AOT;
# native out ops remain the inexpensive implementation inside this boundary.
@torch.library.custom_op('my_cuda_ep::pack',mutates_args=())
def pack(x:torch.Tensor,ids:torch.Tensor,perm:torch.Tensor,inverse:torch.Tensor,topk:int,experts:int)->tuple[torch.Tensor,torch.Tensor]:
    out=torch.empty((ids.numel(),x.shape[1]),device=x.device,dtype=x.dtype);local=torch.empty_like(ids)
    torch.ops.my_cuda_ep.pack_out(x.detach(),ids,perm,topk,experts,out,local)
    return out,local
@pack.register_fake
def _(x,ids,perm,inverse,topk,experts):
    return torch.empty((ids.numel(),x.shape[1]),device=x.device,dtype=x.dtype),torch.empty_like(ids)
def pack_setup(ctx,inputs,output):
    x,ids,perm,inverse,topk,experts=inputs;ctx.save_for_backward(inverse);ctx.shape=x.shape;ctx.topk=topk;ctx.mark_non_differentiable(output[1])
def pack_backward(ctx,grad,grad_local):
    inverse,=ctx.saved_tensors
    index=torch.arange(ctx.shape[0],device=grad.device).repeat_interleave(ctx.topk)
    # Original x[token_index][perm] first restores assignment order in backward.
    dx=torch.zeros(ctx.shape,device=grad.device,dtype=grad.dtype).index_put_((index,),grad[inverse],accumulate=True)
    return dx,None,None,None,None,None
pack.register_autograd(pack_backward,setup_context=pack_setup)

@torch.library.custom_op('my_cuda_ep::gather',mutates_args=())
def gather(x:torch.Tensor,mapping:torch.Tensor)->torch.Tensor:
    out=torch.empty((mapping.numel(),x.shape[1]),device=x.device,dtype=x.dtype)
    torch.ops.my_cuda_ep.gather_out(x.detach(),mapping,out)
    return out
@gather.register_fake
def _(x,mapping):return torch.empty((mapping.numel(),x.shape[1]),device=x.device,dtype=x.dtype)
def gather_setup(ctx,inputs,output):
    x,mapping=inputs;ctx.save_for_backward(mapping);ctx.shape=x.shape
def gather_backward(ctx,grad):
    mapping,=ctx.saved_tensors
    return torch.zeros(ctx.shape,device=grad.device,dtype=grad.dtype).index_copy_(0,mapping,grad),None
gather.register_autograd(gather_backward,setup_context=gather_setup)

_DTYPES=(torch.float32,torch.float16,torch.bfloat16)
@torch.library.custom_op('my_cuda_ep::combine',mutates_args=())
def combine(returned:torch.Tensor,weights:torch.Tensor,inverse:torch.Tensor,topk:int,dtype:int)->torch.Tensor:
    out=torch.empty((inverse.numel()//topk,returned.shape[1]),device=returned.device,dtype=_DTYPES[dtype])
    if topk<=2:
        torch.ops.my_cuda_ep.combine_out(returned.detach(),weights.detach(),inverse,topk,out)
    else:
        weighted=torch.empty((inverse.numel(),returned.shape[1]),device=returned.device,dtype=out.dtype)
        torch.ops.my_cuda_ep.combine_out(returned.detach(),weights.detach(),inverse,1,weighted)
        index=torch.arange(out.shape[0],device=out.device).repeat_interleave(topk)
        out.zero_();out.index_add_(0,index,weighted)
    return out
@combine.register_fake
def _(returned,weights,inverse,topk,dtype):
    return torch.empty((inverse.numel()//topk,returned.shape[1]),device=returned.device,dtype=_DTYPES[dtype])
def combine_setup(ctx,inputs,output):
    returned,weights,inverse,topk,dtype=inputs;ctx.save_for_backward(returned,weights,inverse);ctx.topk=topk
def combine_backward(ctx,grad):
    returned,weights,inverse=ctx.saved_tensors
    upstream=grad.repeat_interleave(ctx.topk,dim=0).to(torch.promote_types(returned.dtype,weights.dtype))
    dy=torch.empty_like(returned).index_copy_(0,inverse,(upstream*weights[:,None]).to(returned.dtype))
    dw=(upstream*returned[inverse]).sum(dim=1).to(weights.dtype)
    return dy,dw,None,None,None
combine.register_autograd(combine_backward,setup_context=combine_setup)
