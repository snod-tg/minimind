"""CPU-only checks for package isolation and canonical native-library lookup."""
import ast
import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from my_cuda_kernels._native import LIBRARIES, repository_root, resolve_library
from script.build import SPECS, build_plan

class PackageContracts(unittest.TestCase):
    def test_imports_have_no_model_dependency(self):
        for file in (repository_root()/'dev/pytorch/my_cuda_kernels').glob('*.py'):
            tree = ast.parse(file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith(('model.', 'minimind')))
            self.assertNotIn('sys.path', file.read_text())
            self.assertNotIn('/data/user4', file.read_text())
        for name in ('ep', 'ep_ops', 'grouped_gemm', 'kernel9', 'native_bf16'):
            importlib.import_module('my_cuda_kernels.'+name)

    def test_resolver_reports_missing_and_accepts_override(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, MY_CUDA_KERNELS_BUILD=tmp):
            with self.assertRaises(FileNotFoundError):
                resolve_library('ep')
            root = Path(tmp)/'ep/current'
            root.mkdir(parents=True)
            library = root/LIBRARIES['ep']
            library.touch()
            self.assertEqual(resolve_library('ep'), library.resolve())
            self.assertEqual(resolve_library('ep', library), library.resolve())

    def test_build_uses_canonical_sources_and_native_precision_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in SPECS:
                out, library, commands, metadata = build_plan(kind, build_root=tmp)
                self.assertFalse(out.exists())
                self.assertEqual(library, LIBRARIES[kind])
                for command in commands[:2]:
                    source = command[command.index('-c')+1]
                    self.assertEqual(Path(source).parent, repository_root()/'dev/cuda')
                self.assertEqual('--fmad=false' in commands[1], kind == 'ep')
                self.assertTrue(metadata['source_sha256'])

if __name__ == '__main__':
    unittest.main()
