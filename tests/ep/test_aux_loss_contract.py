import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn.functional as F
from model.moe_aux_loss import moe_aux_loss


class AuxiliaryContract(unittest.TestCase):
    def test_original_topk_value_and_gradient(self):
        for topk in (1, 2, 3, 6):
            logits = torch.randn(17, 6, requires_grad=True)
            scores = logits.softmax(-1)
            ids = scores.topk(topk).indices
            cfg = SimpleNamespace(num_experts=6, router_aux_loss_coef=5e-4,
                                  router_aux_loss_mode='local_topk_sum')
            ref = (F.one_hot(ids, 6).float().mean(0) * scores.mean(0)).sum()*6*5e-4
            with patch('model.moe_aux_loss.dist.all_reduce') as collective:
                actual = moe_aux_loss(scores, ids, cfg, object())
                collective.assert_not_called()
            self.assertTrue(torch.equal(ref, actual))
            self.assertTrue(torch.equal(torch.autograd.grad(ref, logits, retain_graph=True)[0],
                                        torch.autograd.grad(actual, logits)[0]))

    def test_legacy_paths_unchanged(self):
        scores = torch.tensor([[.6,.3,.1],[.1,.3,.6]])
        ids = torch.tensor([[0,1],[2,1]])
        cfg = SimpleNamespace(num_experts=3, router_aux_loss_coef=.001)
        local = (F.one_hot(ids, 3).float().mean(0)*scores.mean(0)).sum()*3*.001
        self.assertTrue(torch.equal(moe_aux_loss(scores,ids,cfg), local))
        with patch('model.moe_aux_loss.dist.all_reduce'), patch('model.moe_aux_loss.dist.get_world_size', return_value=1):
            global_mean = moe_aux_loss(scores,ids,cfg,object())
        self.assertTrue(torch.equal(global_mean,local/2))
