"""Offline conversion of this measured MiniMind architecture, no source overwrite."""
import argparse,importlib.util,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dev/pytorch'))
from checkpoint_kn import convert_file,decode_state
import torch
p=argparse.ArgumentParser();p.add_argument('--model-source',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True);a=p.parse_args()
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('kn_original_model',a.model_source);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
config=mod.MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=True,use_ep=False,num_experts=4,num_experts_per_tok=1,moe_intermediate_size=2432,dropout=0.,max_position_embeddings=4096)
model=mod.MiniMindForCausalLM(config)
before=hashlib.sha256(Path(a.input).read_bytes()).hexdigest();payload=convert_file(a.input,a.output,model)
original=torch.load(a.input,map_location='cpu',weights_only=True);roundtrip=decode_state(payload)
assert original.keys()==roundtrip.keys() and all(torch.equal(original[k],roundtrip[k]) for k in original)
assert hashlib.sha256(Path(a.input).read_bytes()).hexdigest()==before
report=dict(format=payload['format'],keys=len(original),kn_keys=len(payload['layout']['kn_keys']),linear_weights=len(payload['layout']['linear_keys']),aliases=payload['layout']['aliases'],roundtrip_bitwise=True,source_sha256=before,output_sha256=hashlib.sha256(Path(a.output).read_bytes()).hexdigest())
Path(a.report).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
