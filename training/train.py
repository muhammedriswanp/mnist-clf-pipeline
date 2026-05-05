import os
import torch
import json
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from training.model import MNISTClassifier
from evaluation.plot import plot_curves

BATCH_SIZE = 64 
EPOCHS = 10
LR = 0.001

train_losses, train_accs = [], []
val_losses,   val_accs   = [], []

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = MNISTClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

def train_one_epoch(epoch):
    model.train()
    total_loss, correct = 0, 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_dataset)
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")
    train_losses.append(avg_loss)
    train_accs.append(acc)
    return avg_loss, acc

def validate(epoch):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels).sum().item()

        acc = correct / len(val_dataset)
        avg_loss = total_loss / len(val_dataset)

        print(f"Epoch {epoch} |   Val Loss: {avg_loss:.4f} |   Val Acc: {acc:.4f}")
        val_losses.append(avg_loss)
        val_accs.append(acc)
        return avg_loss, acc
    
def save_model(epoch, val_acc, prev_path=None):
    os.makedirs("models", exist_ok=True)
    
    if prev_path and os.path.exists(prev_path):
        os.remove(prev_path)
        print(f"Deleted old model → {prev_path}")
    
    path = f"models/mnist_model_epoch{epoch}_valacc{val_acc:.4f}.pth"
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")
    return path
        
if __name__ == "__main__":
    best_val_acc = 0.0
    best_model_path = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc     = validate(epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = save_model(epoch, val_acc, best_model_path)

        print("_" * 55)

    plot_curves(train_losses, val_losses, train_accs, val_accs, optimizer_name="SGD")
    print(f"\n✅ Best model: {best_model_path} | Val Acc: {best_val_acc:.4f}")

    # ✅ inside __main__
    os.makedirs("evaluation", exist_ok=True)
    metrics = {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "train_accs":   train_accs,
        "val_accs":     val_accs
    }
    optimizer_name = optimizer.__class__.__name__
    metrics_path = f"evaluation/metrics/metrics_{optimizer_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved → {metrics_path}")