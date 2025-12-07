import torch.nn as nn
import torch

def train_and_evaluate(
    model: nn.Module, 
    train_data: torch.Tensor, 
    train_labels: torch.Tensor, 
    val_data: torch.Tensor, 
    val_labels: torch.Tensor, 
    test_data: torch.Tensor, 
    test_labels: torch.Tensor,
    lr: int,
    batch_size: int,
    epochs: int
    ) -> Tuple[dict, nn.Module]:

    # TODO implement train loop
    # TODO implement val loop
    # TODO implement test loop
    # metrics {} could have acc and loss during training, final acc based on test set
    #return metrics, trained_model
    raise NotImplementedError() 
