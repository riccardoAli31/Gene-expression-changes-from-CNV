"""
Promoter and gene body sepatated convolutional neural network to classify
gene expression changes from mulitome (DNA, ATAC, CNV) data
"""

from torch import nn
import torch

class PromoterCNN(nn.Module):
    """
    This convolutional network is a most simple apporach to the classification
    problem at hand.

    Input dimension: 1..7 x 10_000 per data point
    Output dimension: 1
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.in_dim = hparams.get('in_dim', 7)
        self.out_dim = hparams.get('out_dim', 1)

        # promoter region: 2kb
        # in shape batch_size * 1 * 3 * 2000
        # out shape batchS * 8 * 1 * 660 = (1980 / 3) + 1 = (N + 2 * P - F) / S + 1
        self.conv1pro = nn.Conv2d(1, 8, kernel_size=(self.in_dim, 23), stride=(1, 3))
        self.relu1pro = nn.ReLU()
        self.droppro = nn.Dropout(0.3)
        # out shape 16 * 7 * 105 = (960 / 6) = (N + 2 * P - F) / S
        self.conv2pro = nn.Conv1d(8, 16, kernel_size=36, stride=6)
        self.relu2pro = nn.ReLU()
        # out shape 16 * 7 * 5 = (60 / 8) = (N + 2 * P - F) / S
        self.conv3pro = nn.Conv1d(16, 32, kernel_size=105, stride=1)
        self.relu3pro = nn.ReLU()

        # gene body region: 8kb
        # out shape 16 * 7 * 2660 = (1980 / 2) = (N + 2 * P - F) / S
        self.conv1gen = nn.Conv2d(1, 16, kernel_size=(self.in_dim, 23), stride=(1, 3), padding=(1,0))
        self.relu1gen = nn.ReLU()
        # out shape 16 * 7 * 440 = (2640 / 2) = (N - F) / S
        self.pool1gen = nn.MaxPool2d(kernel_size=(self.in_dim, 26), stride=(1, 6))
        self.relu2gen = nn.ReLU()
        self.dropgen = nn.Dropout(0.3)
        # out shape 32 * 7 * 35 = (420 / 12) = (N + 2 * P - F) / S
        self.conv2gen = nn.Conv1d(16, 32, kernel_size=32, stride=12)
        self.relu3gen = nn.ReLU()
        # out shape 32 * 7 * 5 = (30 / 6) = (N - F) / S
        self.pool2gen = nn.MaxPool1d(kernel_size=35, stride=1)
        self.relu4gen = nn.ReLU()

        # fully connected layer after combination
        self.fc = nn.Linear(32 * 1, self.out_dim)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        # print('batx size: ', x.size())

        x_pro = x[:,:,:,:2000]
        # print('x_pr size: ', x_pro.size())

        # apply convolution to promotor
        promotor_out = self.droppro(self.relu1pro(self.conv1pro(x_pro)))
        promotor_out = promotor_out.squeeze(-2)
        # print('pro1 size: ', promotor_out.size())
        promotor_out = self.relu2pro(self.conv2pro(promotor_out))
        # print('pro2 size: ', promotor_out.size())
        promotor_out = self.relu3pro(self.conv3pro(promotor_out))
        # print('prom size: ', promotor_out.size())

        x_gen = x[:,:,:,2000:]
        # print('x_ge size: ', x_gen.size())

        # apply convolution to gene body
        gene_out = self.relu1gen(self.conv1gen(x_gen))
        # print('gen1 size: ', gene_out.size())
        gene_out = self.dropgen(self.relu2gen(self.pool1gen(gene_out)))
        gene_out = gene_out.squeeze(-2)
        # print('gen2 size: ', gene_out.size())
        gene_out = self.relu3gen(self.conv2gen(gene_out))
        # print('gen3 size: ', gene_out.size())
        gene_out = self.relu4gen(self.pool2gen(gene_out))

        # print('gene size: ', gene_out.size())

        # combine promotor and gene
        output = promotor_out * gene_out
        output = output.squeeze(-1)
        # print('outp size: ', output.size())

        return self.fc(output)
