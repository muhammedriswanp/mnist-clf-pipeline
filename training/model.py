import torch.nn as nn

class MNISTClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256,120),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(120,10)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.network(x)