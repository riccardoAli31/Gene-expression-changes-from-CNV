"""
First implementation of a simple convolutional neural network to classify
gene expression changes from mulitome (DNA, ATAC, CNV) data
"""

from torch import nn

class Simple_CNN(nn.Module):
    """
    This convolutional network is a most simple apporach to the classification
    problem at hand.

    Input dimension: 7x10_000 per data point
    Output dimension: 1
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.in_dim = hparams.get('in_dim', 7 * 10_000)
        self.out_dim = hparams.get('out_dim', 1)

        self.model = nn.Sequential(
            # TODO: reflect dimensions
            nn.Conv1d(1, 8, kernel_size=(7,1000), stride=10),
            # out shape 8 * 7 * 900 = (9000 / 100) = (N + 2 * P - F) / S 
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=(7,100), stride=10),
            # out shape 16 * 7 * 80 = (900 - 100) / 10 = (N + 2 * P - F) / S
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=(7,10), stride=10),
            # out shape 32 * 7 * 7 = (80 - 10) / 10 = (N + 2 * P - F) / S
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, self.out_dim)
        )

    def forward(self, x):
        return self.model(x)
