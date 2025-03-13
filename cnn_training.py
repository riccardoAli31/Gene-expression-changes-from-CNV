"""
Script to train CNN for classification gene expression based on (epi)-genomic data.

Expected usage:
    cnn_training.py <model_name> <batch_nr> <nr_epochs> [optional param]
Optional parameters:
    <include_dna> <include_atac> <include_cnv>

Parameter info:
Name            DataType    Description
model_name      string      name prefix for model
batch_nr        integer     batch number to train on (either 1 or 2)
nr_epochs       integer     number of epochs to train for
include_dna     boolean     if to include DNA data in embeddings
include_atac    boolean     if to include ATAC data in embeddings
include_cnv     boolean     if to include CNV data in embeddings

Hint:
    use pipe operators to save stdout (">") and stderr ("2>")
Example:
    cnn_training.py ... > log/$(date \'+%Y%m%d\').test.batch1.log 2> log/$(date \'+%Y%m%d\')test.batch1.err

author: marcus.wagner@tum.de
"""

# command line asserts TODO: use argparse
import sys
if len(sys.argv) < 4:
    raise RuntimeError('\n'.join(['Too few parameters!', __doc__]))
elif len(sys.argv) > 7:
    raise RuntimeError('\n'.join(['Too many parameters!', __doc__]))
print(sys.argv)

# global variables
from src.data.dataset import CnvDataset
MODEL_NAME = sys.argv[1]
BATCH = sys.argv[2]
EPOCHS = int(sys.argv[3])
INCLUDE_DNA = bool(sys.argv[4]) if len(sys.argv) > 4 else True
INCLUDE_ATAC = bool(sys.argv[5]) if len(sys.argv) > 5 else True
INCLUDE_CNV = bool(sys.argv[6]) if len(sys.argv) > 6 else True
SEQ_LEN = 10_000
IN_DIM = len(CnvDataset._subset_embedding_rows(
    dna=INCLUDE_DNA, atac=INCLUDE_ATAC, cnv=INCLUDE_CNV
    ))
OUT_DIM = 1
print('CNN training script running for batch {}'.format(BATCH))
assert BATCH in ('1', '2'), 'Batch number not known: {}'.format(BATCH)

# paths
from pathlib import Path
git_root = Path('.')
data_root = git_root / 'data'
assert data_root.exists(), \
    'Data directory not found!\n{} does not exist'.format(data_root.absolute())
model_path = git_root / 'model'
assert model_path.is_dir(), 'Directory for saving models does not exist'
plot_path = git_root / 'out' / 'plots' / 'cnn_training'
if not plot_path.exists():
    plot_path.mkdir(parents=True)
tb_log_path = Path('log') / 'tensorboard'
assert tb_log_path.is_dir(), 'Tensorboard logging directory does not exist!'

# read dataset
import pandas as pd
batch_name = 'batch_' + BATCH
train_data_root = data_root / 'embeddings' / batch_name / 'train'
assert train_data_root.is_dir(), \
    'Training data not found: "{}" not a directory'.format(train_data_root)
val_data_root = data_root / 'embeddings' / batch_name / 'val'
assert val_data_root.is_dir(), \
    'Training data not found: "{}" not a directory'.format(val_data_root)

train_df = pd.read_csv(
    data_root / 'splits' / 'batch{}_training_filtered.tsv'.format(BATCH),
    sep='\t'
    )
train_dataset = CnvDataset(
    root=train_data_root, data_df=train_df, include_dna=INCLUDE_DNA,
    include_atac=INCLUDE_ATAC, include_cnv=INCLUDE_CNV
    )
print('Batch {} train loaded: {} data points'.format(BATCH, len(train_dataset)))

val_df = pd.read_csv(
    data_root / 'splits' / 'batch{}_val_filtered.tsv'.format(BATCH), sep='\t'
    )
val_dataset = CnvDataset(
    root=val_data_root, data_df=val_df, include_dna=INCLUDE_DNA,
    include_atac=INCLUDE_ATAC, include_cnv=INCLUDE_CNV
    )
print('Batch {} val loaded: {} data points'.format(BATCH, len(val_dataset)))

# hyper parameters
import torch
hparams = {
    'batch_size': 32,
    'epochs': EPOCHS,
    'lr': 1e-3
}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# define dataloaders
from torch.utils.data import DataLoader
batch_size = hparams.get('batch_size', 32)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# define model
from src.network.chromosome_cnn import ChromosomeCNN
cnn_model = ChromosomeCNN(
    input_dim=IN_DIM, seq_len=SEQ_LEN, output_dim=OUT_DIM
    )
model_name = '_'.join([MODEL_NAME, 'batch' + BATCH])
if not INCLUDE_DNA:
    model_name += '_noDNA'
if not INCLUDE_ATAC:
    model_name += '_noATAC'
if not INCLUDE_CNV:
    model_name += '_noCNV'

# define tensorboard logger
from torch.utils.tensorboard import SummaryWriter
run_number = len([
    d for d in tb_log_path.iterdir() \
        if d.is_dir() and d.name.startswith(model_name)
    ])
tb_log_path = tb_log_path / model_name / ('run' + str(run_number))
model_name = '_'.join([model_name, 'run' + str(run_number)])
tb_logger = SummaryWriter(tb_log_path)

# define training function
from src.network.training import train_model
model_dir = model_path / '_'.join([MODEL_NAME, 'batch' + BATCH])
if not model_dir.exists():
    model_dir.mkdir()
avg_val_loss, best_model = train_model(
    model=cnn_model, hparams=hparams, train_loader=train_loader,
    val_loader=val_loader, tb_logger=tb_logger, device=device,
    model_path=model_dir, model_name=model_name, plot_path=plot_path
)

# TODO: run evaluation
test_data_root = data_root / 'embeddings' / batch_name / 'test'
assert test_data_root.is_dir(), \
    'Training data not found: {} not a directory'.format(test_data_root)
# from src.network.evaluation import test_model
# test_loader = DataLoader(b1_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# TODO: plot performance

