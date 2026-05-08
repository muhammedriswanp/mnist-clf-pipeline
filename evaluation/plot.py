import matplotlib.pyplot as plt
import os

def plot_curves(train_losses, val_losses, train_accs, val_accs, optimizer_name):
    os.makedirs("evaluation", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss")
    ax1.set_title(f"Loss Curve ({optimizer_name})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid()
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs,   label="Val Acc")
    ax2.set_title(f"Accuracy Curve ({optimizer_name})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    path = f"evaluation/plots/{optimizer_name}_curves.png"
    plt.savefig(path)
    plt.close()
    print(f"Plot saved → {path}")
