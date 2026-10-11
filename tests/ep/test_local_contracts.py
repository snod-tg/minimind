"""CPU checks for model ownership, checkpoint precision and launch settings."""
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from model.checkpoint_kn import canonicalize_model, decode_state, encode_state, layout_spec, prepare_native_caches
from model.backends.weights import PackedExpertWeights
from scripts.ep.train import DEFAULT_CONFIG, ROOT, command_for, load_config, validate_gpus

class TiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16, bias=False)
        self.head.weight = self.embedding.weight
    def forward(self, x):
        return self.head(self.embedding(x))

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)

class LocalContracts(unittest.TestCase):
    def test_module_boundaries_without_native_library(self):
        with patch('torch.ops.load_library', side_effect=AssertionError('Import must not load CUDA')):
            for name in ('my_cuda_kernels.ep', 'my_cuda_kernels.ep_ops', 'my_cuda_kernels.grouped_gemm',
                         'my_cuda_kernels.kernel9', 'model.ep.routing', 'model.backends.grouped_gemm',
                         'model.ep_routing_backend', 'model.grouped_gemm_backend'):
                importlib.import_module(name)

    def test_kn_roundtrip_and_shared_parameter(self):
        model = TiedModel().eval()
        original = {k: v.clone() for k, v in model.state_dict().items()}
        spec = layout_spec(model)
        payload = encode_state(model.state_dict(), spec)
        decoded = decode_state(payload)
        for key, value in original.items():
            self.assertTrue(torch.equal(value, decoded[key]))
        tokens = torch.tensor([[1, 3, 7]])
        with torch.no_grad():
            expected = model(tokens).clone()
        canonicalize_model(model)
        self.assertIs(model.embedding.weight, model.head.weight)
        self.assertEqual(model.head.weight.shape, (8, 16))
        model.load_state_dict(payload['state_dict'])
        prepare_native_caches(model)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(tokens), expected))
        model.zero_grad()
        model(tokens).sum().backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        self.assertEqual(set(model.state_dict()), set(original))

    def test_cache_order_and_invalidation(self):
        experts = {i: Expert().eval() for i in (0, 1)}
        snapshots = [e.gate_proj.weight.clone() for e in experts.values()]
        cache = PackedExpertWeights(experts, [1, 0], dtype=torch.bfloat16, device='cpu')
        for j, i in enumerate((1, 0)):
            self.assertTrue(torch.equal(cache.get('gate_proj')[j].T, experts[i].gate_proj.weight.to(torch.bfloat16)))
        for e, value in zip(experts.values(), snapshots):
            self.assertTrue(torch.equal(e.gate_proj.weight, value))
        with torch.no_grad():
            experts[0].gate_proj.weight.add_(1)
        with self.assertRaises(RuntimeError):
            cache.get('gate_proj')
        cache = PackedExpertWeights(experts, [0, 1], device='cpu')
        experts[0].train()
        with self.assertRaises(RuntimeError):
            cache.get('gate_proj')
        with self.assertRaises(ValueError):
            PackedExpertWeights(experts, [0, 0], device='cpu')

    def test_tokenizer_and_trainer_argument_paths(self):
        self.assertTrue((ROOT/'model/tokenizer.json').is_file())
        source=(ROOT/'scripts/ep/train.py').read_text()
        self.assertIn('ROOT / "model/tokenizer.json"', source)
        trainer=(ROOT/'trainer/train_pretrain.py').read_text()
        import ast
        tree=ast.parse(trainer)
        flags={arg.value for node in ast.walk(tree) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument'
               for arg in node.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str)}
        config=load_config(DEFAULT_CONFIG)
        command=command_for(config,ROOT/'runs/test_only','test_only',None)
        for flag in command[7:]:
            if flag.startswith('--'):self.assertIn(flag,flags)

    def test_gpu_selection_and_original_training_config(self):
        self.assertEqual(validate_gpus('2,3'), [2, 3])
        self.assertEqual(validate_gpus('4,7'), [4, 7])
        for value in ('0,4', '2,2', '8,9', '2', '-1,0'):
            with self.assertRaises(ValueError):
                validate_gpus(value)
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual((config['num_experts'], config['num_experts_per_tok'], config['epochs']), (6, 2, 2))
        command = command_for(config, ROOT/'runs/test_only', 'test_only', None)
        for flag, value in (('--from_weight', 'none'), ('--from_resume', '0'), ('--use_compile', '1'),
                            ('--ep_routing', 'compact'), ('--batch_size', '32'), ('--max_seq_len', '340')):
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertFalse(any('stage0' in arg for arg in command))
        self.assertIn('--checkpoint_dir', command)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            import json
            path.write_text(json.dumps(dict(config, num_experts=8, num_experts_per_tok=3)))
            self.assertEqual(load_config(path)['num_experts_per_tok'], 3)
            path.write_text(json.dumps(dict(config, from_resume=1)))
            with self.assertRaises(ValueError):
                load_config(path)

if __name__ == '__main__':
    unittest.main()
