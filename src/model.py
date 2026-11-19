import torch
from torch import nn


class MultiTaskMLP(nn.Module):
    """A *trainable shared layer* followed by two task-specific classifiers.

    The shared layer is essential: GradNorm monitors its parameter gradients.
    """

    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.shared_linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fine_head = nn.Linear(hidden_dim, 100)
        self.coarse_head = nn.Linear(hidden_dim, 20)

    def forward(self, features):
        shared = self.activation(self.shared_linear(features))
        return self.fine_head(shared), self.coarse_head(shared)

    def shared_parameters(self):
        # This is the final (and only) trainable shared layer for GradNorm.
        return tuple(self.shared_linear.parameters())
