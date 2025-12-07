import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Dict

def train_and_evaluate(
    model: nn.Module,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    lr: float,
    batch_size: int,
    epochs: int
) -> Tuple[Dict, nn.Module]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dataloaders
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ----- Evaluation -----
        model.eval()
        correct = 0
        total = 0
        test_loss_epoch = 0.0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss_epoch += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        avg_test_loss = test_loss_epoch / len(test_loader)
        accuracy = correct / total

        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)

        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {accuracy:.4f}")

    metrics = {
        "train_loss": train_losses,
        "test_loss": test_losses,
        "test_accuracy": test_accuracies[-1],
    }

    return metrics, model
 
