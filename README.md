# mnist-clf-pipeline

A complete PyTorch ANN training pipeline for MNIST digit classification with experiment tracking, data versioning, and containerization.

---

## Project Structure

```
mnist-clf-pipeline/
├── data/                            # MNIST dataset (tracked by DVC)
├── notebooks/
│   ├── 01_explore_mnist.ipynb       # Dataset exploration
│   ├── 02_numpy_forward_prop.ipynb  # Forward propagation from scratch
│   ├── 03_activation_functions.ipynb
│   └── 04_compare_lr.ipynb          # Learning rate comparison
├── training/
│   ├── model.py                     # ANN architecture
│   └── train.py                     # Training loop + MLflow logging
├── evaluation/
│   ├── plot.py                      # Loss/accuracy curves
│   ├── plots/                       # Saved plots
│   └── metrics/                     # Saved metrics (JSON)
├── models/                          # Best model checkpoints
├── params.yaml                      # Hyperparameters
├── dvc.yaml                         # DVC pipeline
├── Dockerfile                       # Docker containerization
└── requirements-docker.txt          # Docker dependencies
```

---

## Model Architecture

| Layer | Details |
|---|---|
| Input | 784 (28×28 flattened) |
| Hidden 1 | Linear(784→256) + BatchNorm + ReLU + Dropout(0.3) |
| Hidden 2 | Linear(256→120) + BatchNorm + ReLU + Dropout(0.3) |
| Output | Linear(120→10) |

---

## Results

| Optimizer | Val Accuracy | Convergence |
|---|---|---|
| Adam | **98.15%** | Fast |
| SGD | 95.63% | Slow |

---

## Run

```bash
# Local
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m training.train

# Via DVC pipeline
dvc repro

# Via Docker
docker build -t mnist-clf .
docker run mnist-clf
```

---

## Experiment Tracking

```bash
mlflow ui   # open http://localhost:5000
```

## Data Versioning

```bash
dvc pull    # get dataset
dvc repro   # reproduce experiment
```