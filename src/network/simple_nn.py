"""
First implementation of a simple convolutional neural network to classify
gene expression changes from mulitome (DNA, ATAC, CNV) data
"""

from torch import nn
import torch

class SimpleCNN(nn.Module):
    """
    This convolutional network is a most simple apporach to the classification
    problem at hand.

    Input dimension: 7 x 10_000 per data point
    Output dimension: 1
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.in_dim = hparams.get('in_dim', 7)
        self.out_dim = hparams.get('out_dim', 1)

        # self.dropout = nn.Dropout(0.3),
        self.conv1 = nn.Conv1d(self.in_dim, 16, kernel_size=5000, stride=100)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=8, stride=6)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(16 * 8, self.out_dim)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        # output = self.dropout(output)
        output = self.relu1(self.conv1(x))
        output = self.relu2(self.pool(output))
        output = output.reshape(32, 16 * 8)
        return self.sigm(self.fc(output))


class DeepCNN(nn.Module):
    """
    This convolutional network implements a three hidden layer apporach.

    Input dimension: 7 x 10_000 per data point
    Output dimension: 1
    TODO: implement deepCNN
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.in_dim = hparams.get('in_dim', 7)
        self.out_dim = hparams.get('out_dim', 1)

        self.model = nn.Sequential(
            # TODO: reflect dimensions
            nn.Conv1d(self.in_dim, 8, kernel_size=1000, stride=10),
            # out shape 8 * 7 * 900 = (9000 / 10) = (N + 2 * P - F) / S 
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=(7,100), stride=10),
            # out shape 16 * 7 * 80 = (900 - 10) / 10 = (N + 2 * P - F) / S
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=(7,10), stride=10),
            # out shape 32 * 7 * 7 = (80 - 10) / 10 = (N + 2 * P - F) / S
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, self.out_dim)
        )


        # self.model = nn.Sequential(
        #     # TODO: reflect dimensions
        #     nn.Conv1d(self.in_dim, 16, kernel_size=5000, stride=100),
        #     # out shape 16 * 7 * 50 = (5000 / 100) = (N + 2 * P - F) / S 
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.MaxPool1d(kernel_size=8, stride=6),
        #     # nn.Conv1d(16, 32, kernel_size=8, stride=6),
        #     # out shape 32 * 7 * 7 = (50 - 8) / 6 = (N + 2 * P - F) / S
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(32 * 7 * 7, self.out_dim)
        # )

    def forward(self, x):
        return self.model(x)
