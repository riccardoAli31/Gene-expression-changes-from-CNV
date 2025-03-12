from pathlib import Path
from pandas import read_csv
from ..data.dataset import CnvDataset

def recompute_embedding(dataset_root: Path, dataset_df_path: Path, 
                        genome_fasta: Path, gtf_path: Path, overlap_path: Path,
                        epiAneufinder_path: Path):

    assert dataset_root.is_dir(), 'Dataset root directory not found'
    assert dataset_df_path.is_file(), 'Dataset table not found'

    print('Recomputing embeddings for wrong ATAC embeddings in {}'.format(
        dataset_df_path
    ))

    # 1. load dataset
    dataset = CnvDataset(
        root=dataset_root,
        data_df=read_csv(dataset_df_path, sep='\t')
    )
    print('Loaded dataset:', dataset)

    # 2. filter wrong embeddings
    print('Searching wrong embeddings...')
    recompute_df = dataset.data_df[dataset.data_df.apply(
        lambda r: bool(dataset._load_embedding(r['embedding_path'])[4,:].sum() == 10_000),
        axis=1
    )]
    # recompute_df.apply(lambda r: r['embedding_path'].unlink(), axis=1)
    recompute_df.to_csv(
        dataset_df_path.parent / (dataset_df_path.name + 'redo.tsv'),
        sep='\t'
    )
    recompute_df = recompute_df.drop(columns=['embedding_path'])

    # 3. recompute embeddings
    print('Recomputing embeddings...')
    dataset = CnvDataset(
        root=dataset_root,
        data_df=recompute_df,
        fasta_path=genome_fasta,
        gtf_path=gtf_path,
        atac_path=overlap_path,
        cnv_path=epiAneufinder_path,
        embedding_mode='single_gene_barcode',
        force_recompute=True,
        file_format='pt',
        # verbose=3
    )

