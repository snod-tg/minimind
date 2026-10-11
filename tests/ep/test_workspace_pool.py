import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from model.ep.routing import MiniMindEPRouting


class PoolScope(unittest.TestCase):
    def test_inference_layers_share_scratch_but_legacy_does_not(self):
        controller=MiniMindEPRouting.__new__(MiniMindEPRouting)
        controller.workspaces={};controller.mode='compact'
        x=torch.zeros(2,3)
        with patch('torch.cuda.current_stream',return_value=SimpleNamespace(cuda_stream=1)):
            a=controller.workspace(SimpleNamespace(_ep_routing_layer=0),x)
            b=controller.workspace(SimpleNamespace(_ep_routing_layer=7),x)
            self.assertIs(a,b)
            controller.mode='compact_legacy'
            self.assertIsNot(controller.workspace(SimpleNamespace(_ep_routing_layer=0),x),
                             controller.workspace(SimpleNamespace(_ep_routing_layer=7),x))
        controller.mode='compact'
        with patch('torch.cuda.current_stream',return_value=SimpleNamespace(cuda_stream=2)):
            self.assertIsNot(a,controller.workspace(SimpleNamespace(_ep_routing_layer=0),x))


if __name__=='__main__':unittest.main()
