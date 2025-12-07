import torch
import torch.nn as nn

class MLPProbe(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

