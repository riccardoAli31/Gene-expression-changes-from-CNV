#! /usr/bin/python

"""
Script to compute embeddings for batch 1 splits.
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append('..')
from src.data.dataset import CnvDataset

# file paths
git_root = Path('..')
data_root = git_root / 'data'
out_root = git_root / 'out'
assert data_root.exists()
genome_fasta = data_root / 'reference' / 'GRCh38.d1.vd1.fa'
assert genome_fasta.exists()
overlap_path = data_root / 'overlap_genes_peaks.tsv'
assert overlap_path.exists()
epiAneufinder_path = out_root / 'epiAneufinder' / 'epiAneuFinder_results.tsv'
assert epiAneufinder_path.exists()

# batch 1 train
b1_train_path = data_root / 'splits' / 'batch1_training_filtered.tsv'
df = pd.read_csv(b1_train_path, sep='\t')
dataset = CnvDataset(
    root=data_root / 'embeddings' / 'batch_1' / 'training',
    data_df=df,
    fasta_path=genome_fasta,
    atac_path=overlap_path,
    cnv_path=epiAneufinder_path,
    embedding_mode='single_gene_barcode'
)
print(dataset.data_df)

# batch 1 validation
b1_val_path = data_root / 'splits' / 'batch1_val_filtered.tsv'
df = pd.read_csv(b1_val_path, sep='\t')
dataset = CnvDataset(
    root=data_root / 'embeddings' / 'batch_1' / 'validation',
    data_df=df,
    fasta_path=genome_fasta,
    atac_path=overlap_path,
    cnv_path=epiAneufinder_path,
    embedding_mode='single_gene_barcode'
)
print(dataset.data_df)

# batch 1 test
b1_test_path = data_root / 'splits' / 'batch1_test_filtered.tsv'
df = pd.read_csv(b1_test_path, sep='\t')
dataset = CnvDataset(
    root=data_root / 'embeddings' / 'batch_1' / 'test',
    data_df=df,
    fasta_path=genome_fasta,
    atac_path=overlap_path,
    cnv_path=epiAneufinder_path,
    embedding_mode='single_gene_barcode'
)
print(dataset.data_df)
