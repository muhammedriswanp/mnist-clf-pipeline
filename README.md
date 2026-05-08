# mnist-clf-pipeline

A reusable PyTorch training pipeline for MNIST handwritten digit classification.

## Overview
Trains a neural network classifier on the MNIST dataset using two optimizers (Adam and SGD), with dropout regularization, metric tracking, and model checkpointing.

## Project Structure

```
mnist-clf-pipeline/
├── data/                   # MNIST dataset (auto-downloaded)
├── models/                 # Saved model checkpoints
├── training/
│   ├── model.py            # Neural network architecture
│   └── train.py            # Training and validation loops
├── evaluation/
│   ├── plots/              # Loss and accuracy curves
│   ├── metrics/            # Training metrics (JSON)
│   └── plot.py             # Plotting functions
└── configs/                # Configuration files
```
---

## Model Architecture
| Layer | Details |
|---|---|
| Input | 784 neurons (28×28 flattened) |
| Hidden 1 | 256 neurons + ReLU + Dropout(0.3) |
| Hidden 2 | 120 neurons + ReLU + Dropout(0.3) |
| Output | 10 neurons (digits 0-9) |

**Why this architecture?**
Simple fully connected network — sufficient for MNIST without needing convolutions. Dropout(0.3) added to reduce overfitting.

## Results

| Optimizer | Final Val Accuracy | Convergence |
|---|---|---|
| Adam | 97.84% | Fast |
| SGD (momentum=0.9) | 95.73% | Slow |

**Adam** converged significantly faster — reaching 95% by epoch 1 vs epoch 6 for SGD.
**SGD** is still improving at epoch 10, suggesting it needs more epochs to fully converge.

## Observations
- **No overfitting observed** — train and val accuracy tracked closely throughout
- **Dropout helped** — regularized the model without hurting accuracy
- **Adam outperformed SGD** — better suited for this network size and learning rate

## Run
```bash
# Create environment
python -m venv venv
venv\Scripts\activate
python -m ensurepip --upgrade
python -m pip install torch torchvision matplotlib

# Train
python -m training.train
```

## Output
- Best model saved to `models/`
- Loss/accuracy curves saved to `evaluation/plots/`
- Training metrics saved to `evaluation/metrics/`