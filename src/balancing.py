import torch
from torch import nn


class UncertaintyWeighting(nn.Module):
    """Kendall et al. classification approximation, applied to two CE losses.

    J = sum_i [exp(-s_i) * CE_i + 0.5 * s_i],  s_i=log(sigma_i^2).
    This is the paper's classification APPROXIMATION, not exact temperature NLL.
    """

    def __init__(self, n_tasks=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        losses = torch.stack(list(losses))
        return (torch.exp(-self.log_vars) * losses + 0.5 * self.log_vars).sum()

    @torch.no_grad()
    def weights(self):
        # Effective multipliers on the individual classification CE losses.
        return torch.exp(-self.log_vars.detach()).clone()


class GradNormBalancer(nn.Module):
    """Chen et al. GradNorm with independent network and task-weight objectives.

    The target gradient norms are detached from autograd, exactly as required.
    The network MUST use a detached snapshot of the task weights.
    """

    def __init__(self, n_tasks=2):
        super().__init__()
        self.weights_param = nn.Parameter(torch.ones(n_tasks))

    def grad_loss(self, losses, shared_parameters, initial_losses, alpha=1.5, eps=1e-8):
        loss_vec = torch.stack(list(losses))
        if len(loss_vec) != len(self.weights_param):
            raise ValueError("Incorrect number of GradNorm task losses.")
        if initial_losses is None:
            raise ValueError("GradNorm requires the unweighted initial loss per task.")
        params = tuple(shared_parameters)
        if not params:
            raise ValueError("GradNorm requires trainable shared parameters.")

        unweighted_norms = []
        for loss in losses:
            grads = torch.autograd.grad(
                loss, params, retain_graph=True, create_graph=False, allow_unused=False
            )
            norm = torch.sqrt(
                sum(g.detach().float().square().sum() for g in grads)
                + loss_vec.new_tensor(eps ** 2)
            )
            unweighted_norms.append(norm)
        raw_norms = torch.stack(unweighted_norms).detach()

        # The only gradient path in grad_loss is through weights_param.
        weighted_norms = self.weights_param * raw_norms
        relative = (
            loss_vec.detach().clamp_min(eps)
            / initial_losses.detach().clamp_min(eps)
        )
        inverse_rates = relative / relative.mean().clamp_min(eps)
        target = (
            weighted_norms.detach().mean() * inverse_rates.detach().pow(alpha)
        ).detach()
        auxiliary = (weighted_norms - target).abs().sum()
        diagnostic = {
            "weighted_norms": weighted_norms.detach().cpu().tolist(),
            "unweighted_norms": raw_norms.detach().cpu().tolist(),
            "inverse_rates": inverse_rates.detach().cpu().tolist(),
            "targets": target.detach().cpu().tolist(),
        }
        return auxiliary, diagnostic

    @torch.no_grad()
    def normalize_(self, minimum=1e-4):
        self.weights_param.clamp_(min=minimum)
        self.weights_param.mul_(
            float(self.weights_param.numel()) / self.weights_param.sum()
        )
        if not torch.isfinite(self.weights_param).all():
            raise FloatingPointError("GradNorm weights became non-finite.")

    @torch.no_grad()
    def weights(self):
        return self.weights_param.detach().clone()
