import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        # Simple linear mapping from embedding → logits
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

