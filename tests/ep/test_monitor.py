"""GPU monitor must reject foreign runs and remember its exiting workers."""
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.ep.benchmark_training import processes


class Monitor(unittest.TestCase):
    def test_worker_exit_and_foreign_run(self):
        query="uuid, 123, 1024 MiB\n"
        with patch('subprocess.check_output',return_value=query):
            with patch.object(Path,'read_bytes',return_value=b'python -m scripts.ep.benchmark_training --output /run/ours'):
                self.assertTrue(processes({'uuid'},'/run/ours')[0]['own'])
            with patch.object(Path,'read_bytes',side_effect=FileNotFoundError()):
                self.assertTrue(processes({'uuid'},'/run/ours',{('uuid',123)})[0]['own'])
                self.assertFalse(processes({'uuid'},'/run/ours',set())[0]['own'])
            with patch.object(Path,'read_bytes',return_value=b'python -m scripts.ep.benchmark_training --output /run/foreign'):
                self.assertFalse(processes({'uuid'},'/run/ours',{('uuid',123)})[0]['own'])


if __name__=='__main__':unittest.main()
