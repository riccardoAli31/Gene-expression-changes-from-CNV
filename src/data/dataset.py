import torch
from src.dataloader.embedding import embed
from pathlib import Path
import pandas as pd
from typing import Union, Tuple, List
from tqdm import tqdm


def create_tqdm_bar(iterable, total, desc):
    return tqdm(enumerate(iterable),total=total, ncols=150, desc=desc)


class CopyNumerDNADataset(torch.utils.data.Dataset):
    """
    Dataset class for DNA, ATAC and CNV data.
    """

    def __init__(self, root, data_df: pd.DataFrame, *args,
                 force_recompute=False, embedding_mode='single_gene_barcode',
                 verbose=True, **kwargs):
        """
        Initialization funciton.
        Computes embeddings from raw data, if needed. In this case use kwargs:
        * fasta_path: str as path to reference genome fasta file
        * atac_path: str as path to peaks file from ATAC-seq
        * cnv_path: str as path to EpiAneufinder results

        root : str as data root path
        data_df : pandas.DataFrame with columns:
            'barcode', 'gene_id', 'expression_count', 'classification'
            where 'barcode' and 'gene_id' represent the cell barcode and ENSEMBL
            id of the respective data point while 'expression_count' and
            'classification' are the regression and classification targets. 
        """

        self.root_path = Path(root) / embedding_mode
        self.data_df = data_df

        recompute = force_recompute or not self.root_path.exists()

        barcode_ids = set(data_df['barcode'])
        gene_ids = set(data_df['gene_id'])
        if verbose:
            print('Using {} barcode IDs:'.format(len(barcode_ids)))
            print(','.join(barcode_ids))
            print('Using {} genes IDs:'.format(len(gene_ids)))
            print(','.join(gene_ids))

        if not self.root_path.exists():
            self.root_path.mkdir(parents=True)

        print('Recomputing embeddings: ', recompute)
        if recompute:
            fasta_path = kwargs.get('fasta_path')
            atac_path = kwargs.get('atac_path')
            cnv_path = kwargs.get('cnv_path')

            embedder = embed(
                fasta_path,
                atac_path,
                cnv_path,
                gene_set=gene_ids,
                barcode_set=barcode_ids,
                mode=embedding_mode
            )

            file_paths = list()
            i = 0
            # TODO: make case dependent
            n_embeddings = len(gene_ids) * len(barcode_ids)
            for i, (barcode, gene_id, embedding) in create_tqdm_bar(
                embedder,
                total=n_embeddings,
                desc=f'Computing embedding [{i + 1}/{n_embeddings}]'
            ):
                print('{}: {} embedding for {}, {}'.format(
                    i, embedding.shape, barcode, gene_id
                ))
                i += 1
                match embedding_mode:
                    case 'single_gene_barcode':
                        file_dir = self.root_path / barcode
                        if not file_dir.exists():
                            file_dir.mkdir(parents=True)
                        
                        torch.save(
                            torch.from_numpy(embedding), 
                            file_dir / (gene_id + '.pt')
                        )
                        file_paths.append(
                            file_dir / (gene_id + '.pt')
                        )
                    case 'gene_concat':
                        file_dir = self.root_path
                        if not file_dir.exists():
                            file_dir.mkdir(parents=True)
                        
                        torch.save(
                            torch.from_numpy(embedding), 
                            file_dir / (barcode + '.pt')
                        )
                        file_paths.append(
                            file_dir / (barcode + '.pt')
                        )
                    case 'barcode_channel':
                        file_dir = self.root_path / barcode
                        if not file_dir.exists():
                            file_dir.mkdir(parents=True)

                        torch.save(
                            torch.from_numpy(embedding), 
                            file_dir / (gene_id + '.pt')
                        )
                        file_paths.append(
                            file_dir / (gene_id + '.pt')
                        )

            print(len(file_paths))
            print(self.data_df.shape)
            self.data_df['embedding_path'] = file_paths
        
        # # create table with files
        # match embedding_mode:
        #     case 'single_gene_barcode':
        #         self.data_df['embedding_path'] = [
        #                     self.root_path / barcode / gene + '.pt'
        #                     for gene in gene_ids
        #                 ]
        #     case 'gene_concat':
        #         self.data_df['embedding_path'] = [
        #                     self.root_path / cell + '.pt'
        #                     for cell in barcode_ids
        #                 ]
        #     case 'barcode_channel':
        #         self.data_df['embedding_path'] = [
        #                     self.root_path / gene + '.pt'
        #                     for gene in gene_ids
        #                 ]        
        # # TODO: 
        # # * add labels
        # # * pre-load embeddings from file to buffer I/O time
        # # file size of one single_gene_barcode matrix: 561463 bytes

    @staticmethod
    def _subset_embedding_rows(n_rows: int, dna=True, atac=True, cnv=True):
        """
        Create a list to filter rows of an embedding.
        
        n_rows : int of number of rows of an embedding tensor
        dna : bool if include DNA sequence coding rows, that is rows 0 to 3
        atac : bool if include open chromatin row (4)
        cnv : bool if include CNV encoding rows (5 and 6)
        """

        rows = list()
        if dna:
            rows.extend([0, 1, 2, 3])
        if atac:
            rows.extend([4])
        if cnv:
            rows.extend([5, 6])
        
        # sanity check rows are valid
        assert all(map(lambda x: x < n_rows, rows))

        return rows

    @staticmethod
    def _get_embedding(data_df, idx, rows=Union[List[int], None]):
        """
        Loads embeddings from file and filters rows.

        data_df : pandas.DataFrame with column 'embedding_path'
        idx : int of the embedding to load
        rows : None or List[int] of row indices to include from embedding. This
            parameter is not sanity checked. Make sure to create this list with
            the _subset_embedding_rows() function.
        """

        embedding = torch.load(data_df.loc[idx]['embedding_path'])

        if rows is not None:
            return embedding[rows,:]
        return embedding

    @staticmethod
    def _get_grund_truth_label(data_df: pd.DataFrame, idx: int,
                               type='classification'):
        """
        Return the ground truth respective to regression or classification.

        data_df : pandas.DataFrame with columns:
            'barcode', 'gene_id', 'expression_count', 'classification'
            where 'barcode' and 'gene_id' represent the cell barcode and ENSEMBL
            id of the respective data point while 'expression_count' and
            'classification' are the regression and classification targets.
        idx : int of index in data_df
        type : str, one of 'regression' or 'classification'
        """
        match type:
            case 'classification':
                return data_df.loc[idx]['classification']
            case 'expression_count':
                return data_df.loc[idx]['expression_count']

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx, **kwargs):
        return {
            'embedding': self._get_embedding(self.data_df, idx, **kwargs),
            'label': self._get_grund_truth_label(self.data_df, idx)
        }

    def split(self, train, test, val=None):
        """
        Function to create training, validation and test splits.
        """
        # TODO: write function to make test/val/train splits
        pass
                
