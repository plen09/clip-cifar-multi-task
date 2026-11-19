"""Unit tests run without downloading CIFAR or CLIP; use python -m unittest."""
import unittest

import torch
import torch.nn.functional as F

from src.balancing import GradNormBalancer, UncertaintyWeighting
from src.data import stratified_split
from src.model import MultiTaskMLP


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_uncertainty_classification_formula_and_grad(self):
        uw = UncertaintyWeighting(2)
        losses = [torch.tensor(2.0), torch.tensor(3.0)]
        result = uw(losses)
        self.assertAlmostEqual(result.item(), 5.0)
        result.backward()
        self.assertTrue(torch.allclose(
            uw.log_vars.grad, torch.tensor([-1.5, -2.5]), atol=1e-5
        ))

    def test_shared_layer_receives_both_task_gradients(self):
        model = MultiTaskMLP()
        x = torch.randn(16, 512)
        fine, coarse = model(x)
        losses = [F.cross_entropy(fine, torch.randint(0, 100, (16,))),
                  F.cross_entropy(coarse, torch.randint(0, 20, (16,)))]
        for loss in losses:
            grad = torch.autograd.grad(loss, model.shared_linear.weight, retain_graph=True)[0]
            self.assertGreater(float(grad.norm()), 0.0)

    def test_gradnorm_target_is_detached_and_weights_normalize(self):
        model = MultiTaskMLP()
        balancer = GradNormBalancer()
        x = torch.randn(32, 512)
        fine, coarse = model(x)
        losses = [F.cross_entropy(fine, torch.randint(0, 100, (32,))),
                  F.cross_entropy(coarse, torch.randint(0, 20, (32,)))]
        l0 = torch.stack(losses).detach() * torch.tensor([1.2, 0.7])
        aux, diag = balancer.grad_loss(losses, model.shared_parameters(), l0)
        self.assertEqual(len(diag["targets"]), 2)
        self.assertFalse(any(torch.isnan(torch.tensor(diag["targets"]))))
        aux.backward()
        self.assertIsNotNone(balancer.weights_param.grad)
        self.assertTrue(torch.isfinite(balancer.weights_param.grad).all())
        # Aux optimization must not backpropagate through the shared network.
        self.assertIsNone(model.shared_linear.weight.grad)
        opt = torch.optim.Adam([balancer.weights_param], lr=0.01)
        opt.step()
        balancer.normalize_()
        self.assertAlmostEqual(float(balancer.weights().sum()), 2.0, places=5)
        self.assertTrue(torch.all(balancer.weights() > 0))

    def test_stratified_split(self):
        labels = torch.arange(100).repeat_interleave(10)
        tr, val = stratified_split(labels, 0.1, seed=2026)
        self.assertEqual(len(tr), 900)
        self.assertEqual(len(val), 100)
        self.assertEqual(len(set(tr.tolist()) & set(val.tolist())), 0)
        self.assertTrue(torch.equal(torch.bincount(labels[val]), torch.ones(100, dtype=torch.long)))
        tr2, val2 = stratified_split(labels, 0.1, seed=2026)
        self.assertTrue(torch.equal(val, val2))


if __name__ == "__main__":
    unittest.main()
