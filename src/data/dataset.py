#!/ust/bin/python
from typing import Union, List, Dict
from pathlib import Path
from pandas import DataFrame, merge
from numpy import ndarray, uint8
from torch import (
    Tensor,
    save as pyt_save,
    load as pyt_load,
    from_numpy as pyt_from_numpy
)
from torch.utils.data import Dataset
from scipy.io import mmread, mmwrite
from .embedding import embed
import gzip
from warnings import warn


# TODO:
# * add labels
# * pre-load embeddings from file to buffer I/O time
# file size of one single_gene_barcode matrix: 561463 bytes


class CnvDataset(Dataset):
    """
    Dataset class for DNA, ATAC and CNV data.
    """

    def __init__(self, root, data_df: DataFrame, *args,
                 force_recompute=False, embedding_mode='single_gene_barcode',
                 file_format='mtx', use_gzip=False, verbose=1, 
                 dtype=uint8, target_type='classification', **kwargs):
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

        TODO:
        * get embedding file paths by traversing file tree
        * debugging embedding creation
        * update documentation 
        """

        assert not data_df.empty, 'DataFrame data_df is empty!'
        assert embedding_mode in (
            'single_gene_barcode', 'gene_concat', 'barcode_channel'
        ), 'Unsupported embedding mode: {}'.format(embedding_mode)
        assert file_format in ('pt', 'mtx'), 'File format not supported!'
        assert target_type in ('classification', 'regression'),\
            'Unknown target_type: {} != "classification" / "regression"'.format(
                target_type
            )
        
        super().__init__(*args, **kwargs)

        self.root_path = Path(root) / embedding_mode
        self.data_df = data_df
        self.embedding_mode = embedding_mode
        self.file_format = file_format
        self.compress = use_gzip
        self.dtype = dtype
        if target_type == 'classification':
            self.target_type = target_type
        elif target_type == 'regression':
            self.target_type = 'expression_count'
        else:
            raise RuntimeError('Unknown target type: {}'.format(target_type))

        recompute = force_recompute or not self.root_path.exists()

        barcode_ids = set(data_df['barcode'])
        gene_ids = set(data_df['gene_id'])
        if verbose > 0:
            print('Using {} barcodes'.format(len(barcode_ids)))
            print(','.join(barcode_ids)) if verbose > 2 else None
            print('Using {} genes'.format(len(gene_ids)))
            print(','.join(gene_ids)) if verbose > 2 else None

        if not self.root_path.exists():
            self.root_path.mkdir(parents=True)

        # define merge_df variable here to overwrite later
        merge_df = None

        if recompute:
            if verbose > 0:
                print('Recomputing embeddings: ', recompute)
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

            path_list = list()
            barcode_list = list()
            gene_id_list = list()
            for barcode, gene_id, embedding in embedder:
                if verbose > 2:
                    print('{} embedding for {}, {}'.format(
                        embedding.shape, barcode, gene_id
                    ))
                f_path = self.ids_to_emb_path(barcode, gene_id)
                self._save_embedding(embedding=embedding, file_path=f_path)
                barcode_list.append(barcode)
                gene_id_list.append(gene_id)
                path_list.append(f_path)
            
            # filter df for samples with embeddings
            emb_df = DataFrame({
                'barcode': barcode_list,
                'gene_id': gene_id_list,
                'embedding_path': path_list
            })

            if verbose > 1:
                print('emb_df\n', emb_df)

            merge_df = merge(
                self.data_df,
                emb_df,
                how='right',
                on=['barcode', 'gene_id']
            )
        
        else:
            # filter data points by file paths from directory traversion
            file_list = list()
            if embedding_mode == 'single_gene_barcode':
                file_list = [
                    f for d in self.root_path.iterdir() for f in d.iterdir()
                    if f.name.split('.')[-1] == self.file_format
                ]
            else:
                file_list = [
                    f for f in self.root_path.iterdir()
                    if f.name.split('.')[-1] == self.file_format
                ]

            # create dataframe with file found on disk
            record_list = [self.emb_path_to_ids(p) for p in file_list]
            emb_on_disk_df = DataFrame.from_records(record_list)
            emb_on_disk_df['embedding_path'] = file_list

            if verbose > 1:
                print('emb_on_disk_df')
                print(emb_on_disk_df)
                print('CnVDataset.data_df')
                print(self.data_df)

            # merge data
            merge_df = merge(
                self.data_df,
                emb_on_disk_df,
                how='outer',
                on=['barcode', 'gene_id']
            )

        # report missing/unused embeddings for both cases
        if verbose > 1:
            print('merge_df')
            print(merge_df)

        # print missing
        missing_df = merge_df[merge_df['embedding_path'].isna()]
        if missing_df.shape[0] > 0:
            print('No embedding files for {} data points in {}!'.format(
                missing_df.shape[0], self.root_path
            ))
            if verbose > 2:
                print(missing_df)
        elif missing_df.shape[0] == self.data_df.shape[0]:
            raise RuntimeError('No embedding files found!')
        
        # print unused
        unused_df = merge_df[merge_df[self.target_type].isna()]
        if unused_df.shape[0] > 0:
            print('Found {} embedding files with no target value in {}!'.format(
                unused_df.shape[0], self.root_path
            ))
            if verbose > 2:
                print(unused_df)
        
        self.data_df = merge_df.loc[
            merge_df['embedding_path'].isin(file_list) &
            ~merge_df[self.target_type].isna()
        ]

    def ids_to_emb_path(self, barcode: str, gene_id: str, mkdir=False) -> Path:
        """
        Return path to an embedding, depending on the embedding mode.
        """
        
        file_dir = self.root_path
        file_name = ''

        match self.embedding_mode:
            case 'single_gene_barcode':
                file_dir = self.root_path / barcode
                file_name = gene_id + '.' + self.file_format
            case 'gene_concat':
                file_name = barcode + '.' + self.file_format
            case 'barcode_channel':
                file_name = gene_id + '.' + self.file_format
        
        if mkdir and not file_dir.exists():
            file_dir.mkdir(parents=True)

        return file_dir / file_name
    
    def emb_path_to_ids(self, emb_path: Path) -> Dict[str, str]:
        """
        Convert an emedding file path to associated ids.
        """
        file_name_id = emb_path.name.split('.')[0]
        match self.embedding_mode:
            case 'single_gene_barcode':
                barcode = emb_path.parts[-2]
                return {'barcode': barcode, 'gene_id': file_name_id}
            case 'gene_concat':
                return {'barode': file_name_id}
            case 'barcode_channel':
                return {'gene_id': file_name_id}

    @staticmethod
    def _save_embedding(embedding: ndarray, file_path: Path, 
                        compress:bool=False):
        """
        Save embedding to file. Supported formats: '.pt' and '.mtx' both with
        gzip compression.
        """

        file_format = file_path.name.split('.')[-1]
        file = file_path
        if compress:
            file_path = file_path.parent / file_path.name + '.gz'
            file = gzip.open(file_path, 'wb')

        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True)

        match file_format:
            case 'pt':
                pyt_save(pyt_from_numpy(embedding), file)
            case 'mtx':
                mmwrite(file, embedding)
    
    @staticmethod
    def _load_embedding(file_path: Path, dtype=uint8) -> Tensor:
        """
        Load an embedding from file depending on the file format.
        """

        file_format = file_path.name.split('.')[-1]
        if file_format == 'gz':
            file_path = gzip.open(file_path)
            file_format = file_path.name.split('.')[-2]

        match file_format:
            case 'pt':
                return pyt_load(file_path)
            case 'mtx':
                return pyt_from_numpy(mmread(file_path).astype(dtype))
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
    def _get_embedding(data_df, idx: int, rows:Union[List[int], None]=None):
        """
        Loads embeddings from file and filters rows.

        data_df : pandas.DataFrame with column 'embedding_path'
        idx : int of the embedding to load
        rows : None or List[int] of row indices to include from embedding. This
            parameter is not sanity checked. Make sure to create this list with
            the _subset_embedding_rows() function.
        """

        file_path = data_df.iloc[idx]['embedding_path']
        embedding = CnvDataset._load_embedding(file_path)

        if rows is not None:
            return embedding[rows,:]
        return embedding

    @staticmethod
    def _get_grund_truth(data_df: DataFrame, idx: int, target_type: str):
        """
        Return the ground truth respective to regression or classification.

        data_df : pandas.DataFrame with columns:
            'barcode', 'gene_id', 'expression_count', 'classification'
            where 'barcode' and 'gene_id' represent the cell barcode and ENSEMBL
            id of the respective data point while 'expression_count' and
            'classification' are the regression and classification targets.
        idx : int of index in data_df
        type : str, one of 'expression_count' or 'classification'
        """
        
        return data_df.iloc[idx][target_type]

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx, **kwargs):
        return {
            'embedding': self._get_embedding(self.data_df, idx, **kwargs),
            'target': self._get_grund_truth(self.data_df, idx, self.target_type)
        }

    def __repr__(self):
        return '{} with {} datapoints'.format(self.__class__, len(self))

    def __str__(self):
        return '{} with {} datapoints\n{}'.format(
            self.__class__, len(self), self.data_df.head()
        )

    def class_balance(self):
        assert self.target_type == 'classification', \
            'Only class balance for classification'
        return self.data_df[self.target_type].unique()
        # TODO: count class labels and divide by

    # def split(self, train, test, val=None):
    #     """
    #     Function to create training, validation and test splits.
    #     """
    #     # TODO: write function to make test/val/train splits
    #     pass
                
