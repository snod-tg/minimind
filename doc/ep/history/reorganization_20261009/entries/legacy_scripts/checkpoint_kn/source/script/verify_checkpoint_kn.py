"""Full model default BF16 equivalence, checkpoint roundtrip and training aliases."""
import argparse,importlib.util,sys,json,copy,gc
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dev/pytorch'))
from checkpoint_kn import *
p=argparse.ArgumentParser();p.add_argument('--model-source',required=True);p.add_argument('--original',required=True);p.add_argument('--converted',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261007)
spec=importlib.util.spec_from_file_location('kn_verify_model',a.model_source);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
config=mod.MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True,use_ep=False,num_experts=4,num_experts_per_tok=1,moe_intermediate_size=2432,dropout=0.,max_position_embeddings=4096)
original=mod.MiniMindForCausalLM(config);original.load_state_dict(torch.load(a.original,map_location='cpu',weights_only=True));original=original.cuda().eval()
kn=load_kn_checkpoint(mod.MiniMindForCausalLM(config),a.converted).cuda().eval()
assert kn.lm_head.weight is kn.model.embed_tokens.weight
assert kn.lm_head.weight.shape==(768,6400)
prepare_native_caches(kn);rows=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for topk in [1,2]:
  for layer in original.model.layers:layer.mlp.top_k=topk
  for layer in kn.model.layers:layer.mlp.top_k=topk
  # Actual implementation reads num_experts_per_tok; set both for robustness.
  for model in [original,kn]:
   model.config.num_experts_per_tok=topk
   for layer in model.model.layers:layer.mlp.num_experts_per_tok=topk
  for batch,length in [(1,1),(2,17),(1,128),(8,340),(1,513)]:
   ids=torch.randint(3,6400,(batch,length),device='cuda')
   ref=original(ids,use_cache=True);actual=kn(ids,use_cache=True)
   assert torch.equal(ref.logits,actual.logits),(topk,batch,length)
   assert all(torch.equal(x,y) for r,s in zip(ref.past_key_values,actual.past_key_values) for x,y in zip(r,s))
   rows.append(dict(topk=topk,batch=batch,length=length,bitwise_logits=True,bitwise_kv=True))
   for _ in range(4):
    token=ref.logits[:,-1].argmax(-1,keepdim=True)
    ref=original(token,past_key_values=ref.past_key_values,use_cache=True);actual=kn(token,past_key_values=actual.past_key_values,use_cache=True)
    assert torch.equal(ref.logits,actual.logits)
   rows[-1]['decode_steps_bitwise']=4
   (out/'model.json').write_text(json.dumps(rows,indent=2));print('MODEL',rows[-1],flush=True)
 save_kn_checkpoint(kn,out/'resaved_kn.pth')
 saved=torch.load(out/'resaved_kn.pth',map_location='cpu',weights_only=True)
 assert saved['format']==FORMAT and saved['state_dict']['lm_head.weight'].shape==(768,6400)
 recovered=decode_state(saved);source=torch.load(a.original,map_location='cpu',weights_only=True)
 assert all(torch.equal(source[key],recovered[key]) for key in source)
 # Reload into the canonical model clears old native caches and preserves values.
 load_kn_checkpoint(kn,out/'resaved_kn.pth');assert kn.lm_head.weight is kn.model.embed_tokens.weight
# Meaningful gradient test: shared embedding/head, linear bias, and optimizer update.
class Tiny(nn.Module):
 def __init__(self):
  super().__init__();self.embed=nn.Embedding(11,5);self.hidden=nn.Linear(5,5);self.head=nn.Linear(5,11,bias=False);self.head.weight=self.embed.weight
 def forward(self,ids):return self.head(self.hidden(self.embed(ids)))
native=Tiny().double();canonical=canonicalize_model(copy.deepcopy(native));ids=torch.tensor([[1,2,3],[4,2,1]])
assert canonical.embed.weight is canonical.head.weight
for model in [native,canonical]:model(ids).square().sum().backward()
for key,value in native.named_parameters():
 c=dict(canonical.named_parameters())[key];expected=value.grad.T if key in canonical._kn_layout['kn_keys'] else value.grad
 torch.testing.assert_close(c.grad,expected,atol=1e-10,rtol=1e-10)
for model in [native,canonical]:torch.optim.SGD(model.parameters(),lr=.001).step()
torch.testing.assert_close(canonical(ids),native(ids),atol=1e-10,rtol=1e-10)
# Versioned cache refresh after updating canonical parameters.
with torch.no_grad():
 before=canonical(ids).clone();canonical.hidden.weight.add_(.125);after=canonical(ids)
 expected=F.linear(F.linear(F.embedding(ids,canonical.embed.weight.T.contiguous()),canonical.hidden.weight.T.contiguous(),canonical.hidden.bias),canonical.head.weight.T.contiguous())
 torch.testing.assert_close(after,expected,atol=0,rtol=0)
 assert not torch.equal(before,after)
rejected=0
try:convert_file(a.converted,out/'double_transpose.pth',original)
except ValueError:rejected+=1
try:load_kn_checkpoint(mod.MiniMindForCausalLM(config),a.original)
except ValueError:rejected+=1
assert rejected==2
summary=dict(rejected_formats=rejected,model_cases=len(rows),decode_steps=sum(r['decode_steps_bitwise'] for r in rows),native_bf16_bitwise=True,kv_bitwise=True,roundtrip_bitwise=True,resave_kn=True,alias_preserved=True,training_gradients=True,optimizer_step=True,cache_refresh=True,allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('ALL CHECKS PASSED',summary)
