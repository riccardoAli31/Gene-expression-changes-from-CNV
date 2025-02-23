#! /usr/bin/python

"""
Script to compute all embeddings for batch 1 and 2.
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
b1_full_data = out_root / 'preprocessing' / 'classification_median_batch_1.tsv'
assert b1_full_data.exists()
b2_full_data = out_root / 'preprocessing' / 'classification_median_batch_2.tsv'
assert b2_full_data.exists()
b1_val_path = data_root / 'batch1_val.tsv'
assert b1_val_path.exists()

# compute all embeddings for batch 1
df = pd.read_csv(b1_val_path, sep='\t')
CopyNumerDNADataset(
    root=data_root / 'embeddings' / 'batch_1' / 'val',
    data_df=df,
    fasta_path=genome_fasta,
    atac_path=overlap_path,
    cnv_path=epiAneufinder_path,
    force_recompute=True,
    embedding_mode='single_gene_barcode'
)

# # compute all embeddings for batch 2
# df = pd.read_csv(b2_full_data, sep='\t')
# b1_dataset = CopyNumerDNADataset(
#     root=data_root / 'embeddings' / 'batch_2' ,
#     data_df=df,
#     fasta_path=genome_fasta,
#     atac_path=overlap_path,
#     cnv_path=epiAneufinder_path,
#     force_recompute=True,
#     embedding_mode='single_gene_barcode'
# )

