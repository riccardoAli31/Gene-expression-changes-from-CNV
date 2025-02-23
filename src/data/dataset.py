#!/ust/bin/python
from typing import Union, List
from pathlib import Path
from pandas import DataFrame
from numpy import ndarray
import torch
from scipy.io import mmread, mmwrite
from .embedding import embed


# TODO:
# * implement compression with gzip
# * add labels
# * pre-load embeddings from file to buffer I/O time
# file size of one single_gene_barcode matrix: 561463 bytes


class CnvDataset(torch.utils.data.Dataset):
    """
    Dataset class for DNA, ATAC and CNV data.
    """

    def __init__(self, root, data_df: DataFrame, *args,
                 force_recompute=False, embedding_mode='single_gene_barcode',
                 file_format='mtx', verbose=1, **kwargs):
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
        if verbose > 0:
            print('Using {} barcode IDs:'.format(len(barcode_ids)))
            print(','.join(barcode_ids))
            print('Using {} genes IDs:'.format(len(gene_ids)))
            print(','.join(gene_ids))

        if not self.root_path.exists():
            self.root_path.mkdir(parents=True)

        if verbose > 0:
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
            for barcode, gene_id, embedding in embedder:
                if verbose > 2:
                    print('{} embedding for {}, {}'.format(
                        embedding.shape, barcode, gene_id
                    ))
                match embedding_mode:
                    case 'single_gene_barcode':
                        file_dir = self.root_path / barcode
                        if not file_dir.exists():
                            file_dir.mkdir(parents=True)
                        
                        f_path = self.save_embedding(
                            embedding, file_dir, gene_id, file_format
                        )
                        file_paths.append(f_path)
                    case 'gene_concat':
                        file_dir = self.root_path
                        if not file_dir.exists():
                            file_dir.mkdir(parents=True)
                        
                        f_path = self.save_embedding(
                            embedding, file_dir, barcode, file_format
                        )
                        file_paths.append(f_path)
                    case 'barcode_channel':
                        file_dir = self.root_path / barcode
                        if not file_dir.exists():
                            file_dir.mkdir(parents=True)

                        f_path = self.save_embedding(
                            embedding, file_dir, gene_id, file_format
                        )
                        file_paths.append(f_path)

            print(len(file_paths))
            print(self.data_df.shape)
            self.data_df['embedding_path'] = file_paths

        else:
            # add file path column to self.data_df based on mode
            match embedding_mode:
                case 'single_gene_barcode':
                    self.data_df['embedding_path'] = [
                        self.root_path / barcode / (gene + '.' + file_format)
                        for gene in gene_ids
                    ]
                case 'gene_concat':
                    self.data_df['embedding_path'] = [
                        self.root_path / (cell + '.' + file_format)
                        for cell in barcode_ids
                    ]
                case 'barcode_channel':
                    self.data_df['embedding_path'] = [
                        self.root_path / (gene + '.' + file_format)
                        for gene in gene_ids
                    ]
    
    @staticmethod
    def save_embedding(embedding: ndarray, out_dir: Path, file_name: str,
                        file_format:str='mtx') -> Path:
        """
        Function to save embedding to file.

        TODO: add gzip functionality
        """

        file_path = out_dir / (file_name + '.' + file_format)
        match file_format:
            case 'pt':
                torch.save(torch.from_numpy(embedding), file_path)
            case 'mtx':
                mmwrite(file_path, embedding)
        return file_path
    
    @staticmethod
    def load_embedding(file_path: Path) -> torch.Tensor:
        """
        Load an embedding from file depending on the file format.

        TODO:
        * gunzip for decompression
        """

        file_format = file_path.name.split('.')[-1]
        match file_format:
            case 'pt':
                return torch.load(file_path)
            case 'mtx':
                return torch.from_numpy(mmread(file_path))
        raise RuntimeError('Unsupported file format: {}'.format(file_format))

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

        file_path = data_df.loc[idx]['embedding_path']
        embedding = CnvDataset.load_embedding(file_path)

        if rows is not None:
            return embedding[rows,:]
        return embedding

    @staticmethod
    def _get_grund_truth_label(data_df: DataFrame, idx: int,
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

    def __str__(self):
        return '{} with {} datapoints'.format(self.__class__, len(self))

    # def split(self, train, test, val=None):
    #     """
    #     Function to create training, validation and test splits.
    #     """
    #     # TODO: write function to make test/val/train splits
    #     pass
                
