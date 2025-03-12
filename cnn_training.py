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
from torch.utils.tensorboard import SummaryWriter
import copy
from src.data.dataset import CnvDataset
from src.network.training import train_model, evaluate_model
from src.network.chromosome_cnn import ChromosomeCNN

# paths
git_root = Path('.')
data_root = git_root / 'data'
assert data_root.exists()

# global variables TODO: parse them from command line
BATCH = 'batch_1'
EPOCHS = 20
INCLUDE_DNA = True
INCLUDE_ATAC = True
INCLUDE_CNV = True
IN_DIM = len(CnvDataset._subset_embedding_rows(
    dna=INCLUDE_DNA, atac=INCLUDE_ATAC, cnv=INCLUDE_CNV
    ))
print('Training CNN for {}'.format(BATCH))

# dataset import
dataset_root_train = data_root / 'embeddings' / BATCH / 'train'
dataset_root_val = data_root / 'embeddings' / BATCH / 'val'
dataset_root_test = data_root / 'embeddings' / BATCH / 'test'

b1_train_path = data_root / 'splits' / 'batch1_training_filtered.tsv'
b1_train_df = pd.read_csv(b1_train_path, sep='\t')
b1_train_dataset = CnvDataset(
    root=dataset_root_train, data_df=b1_train_df, include_dna=INCLUDE_DNA,
    include_atac=INCLUDE_ATAC, include_cnv=INCLUDE_CNV
    )
print('{} train loaded: {} data points'.format(BATCH, len(b1_train_dataset)))

b1_val_path = data_root / 'splits' / 'batch1_val_filtered.tsv'
b1_val_df = pd.read_csv(b1_val_path, sep='\t')
b1_val_dataset = CnvDataset(
    root=dataset_root_val, data_df=b1_val_df, include_dna=INCLUDE_DNA,
    include_atac=INCLUDE_ATAC, include_cnv=INCLUDE_CNV
    )
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


# define tensorboard logger
tb_log_path = Path('log') / 'tensorboard'
run_number = len([
    d for d in tb_log_path.iterdir() if d.is_dir() and d.name.startswith('run_')
    ])
tb_log_path = tb_log_path / '_'.join(['run', str(run_number)])
tb_logger = SummaryWriter(tb_log_path)

# define dataloaders
batch_size = hparams.get('batch_size', 32)
train_loader = DataLoader(b1_train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(b1_val_dataset, batch_size=batch_size, shuffle=True)

# define model
cnn_model = ChromosomeCNN(input_dim=IN_DIM, seq_len=10_000, output_dim=1)

# define training function
# TODO: train training
avg_val_loss, best_model = train_model(
    cnn_model, train_loader, val_loader, tb_logger, name='ChromosomeCNN'
    )

# TODO: run evaluation
# test_loader = DataLoader(b1_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
evaluate_model(cnn_model, b1_val_dataset)

# TODO: plot performance

