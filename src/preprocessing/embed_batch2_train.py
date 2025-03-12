#! /usr/bin/python

"""
Script to compute embeddings for batch 2 training split.
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append('../..')
from src.data.dataset import CnvDataset
from src.preprocessing.embedding_redo import recompute_embedding

# file paths
git_root = Path('../..')
data_root = git_root / 'data'
out_root = git_root / 'out'
assert data_root.exists()
genome_fasta = data_root / 'reference' / 'GRCh38.d1.vd1.fa'
assert genome_fasta.exists()
gtf_path = data_root / 'gene_positions_and_overlaps' / 'gene_positions.csv'
assert gtf_path.exists()
overlap_path = data_root / 'gene_positions_and_overlaps' / 'overlaps_batch2.tsv'
assert overlap_path.exists()
epiAneufinder_path = out_root / 'epiAneufinder' / 'epiAneuFinder_results.tsv'
assert epiAneufinder_path.exists()

# batch 2 train
# b2_train_path = data_root / 'splits' / 'batch2_training_filtered.tsv'
# df = pd.read_csv(b2_train_path, sep='\t')
# dataset = CnvDataset(
#     root=data_root / 'embeddings' / 'batch_2' / 'train',
#     data_df=df,
#     fasta_path=genome_fasta,
#     gtf_path=gtf_path,
#     atac_path=overlap_path,
#     cnv_path=epiAneufinder_path,
#     embedding_mode='single_gene_barcode',
#     force_recompute=True,
# )

# recompute wrong ATAC embeddings
recompute_embedding(
    data_root / 'embeddings' / 'batch_2' / 'train',
    data_root / 'splits' / 'batch2_training_filtered.tsv',
    genome_fasta,
    gtf_path,
    overlap_path,
    epiAneufinder_path
)
