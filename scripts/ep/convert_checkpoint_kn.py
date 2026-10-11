"""Convert a full MiniMind model checkpoint to KN without overwriting its source."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.checkpoint_kn import convert_file, decode_state

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='MiniMind model config JSON, not trainer settings')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError('Output checkpoint/report already exists; choose new paths')
    if args.input.resolve() in (args.output.resolve(), args.report.resolve()) or args.output.resolve()==args.report.resolve():
        raise ValueError('Source, output and report must have distinct paths')
    values = json.loads(args.config.read_text())
    values = values.get('model', values)
    values['use_ep'] = False
    model = MiniMindForCausalLM(MiniMindConfig(**values))
    before = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = convert_file(args.input, args.output, model)
    original = torch.load(args.input, map_location='cpu', weights_only=True)
    roundtrip = decode_state(payload)
    assert all(torch.equal(original[k], roundtrip[k]) for k in original)
    assert hashlib.sha256(args.input.read_bytes()).hexdigest()==before
    args.report.write_text(json.dumps(dict(format=payload['format'], roundtrip_bitwise=True,
        keys=len(original), kn_keys=len(payload['layout']['kn_keys']), source_sha256=before,
        output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()), indent=2)+'\n')

if __name__ == '__main__':
    main()
