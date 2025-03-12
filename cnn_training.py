"""
Script to train the CNN for this project.
"""

from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset
from torch.amp import autocast
from torch import nn, optim
import copy
from tqdm import tqdm
from src.data.dataset import CnvDataset
from src.network.training import train_model, evaluate_model

# paths
git_root = Path('.')
data_root = git_root / 'data'
assert data_root.exists()

# global variables TODO: parse them from command line
BATCH = 'batch_1'
EPOCHS = 20
print('Training CNN for {}'.format(BATCH))

# dataset import
dataset_root_train = data_root / 'embeddings' / BATCH / 'train'
dataset_root_val = data_root / 'embeddings' / BATCH / 'val'
dataset_root_test = data_root / 'embeddings' / BATCH / 'test'

b1_train_path = data_root / 'splits' / 'batch1_training_filtered.tsv'
b1_train_df = pd.read_csv(b1_train_path, sep='\t')
b1_train_dataset = CnvDataset(root=dataset_root_train, data_df=b1_train_df)
print('{} train loaded: {} data points'.format(BATCH, len(b1_train_dataset)))

b1_val_path = data_root / 'splits' / 'batch1_val_filtered.tsv'
b1_val_df = pd.read_csv(b1_val_path, sep='\t')
b1_val_dataset = CnvDataset(root=dataset_root_val, data_df=b1_val_df)
print('{} val loaded: {} data points'.format(BATCH, len(b1_val_dataset)))

# hyper parameters
hparams = {
    'batch_size': 32,
    'epochs': EPOCHS,
    'lr': 1e-3
}
sequ_len = 10_000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# define training function


