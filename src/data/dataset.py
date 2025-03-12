#!/ust/bin/python
from typing import Union, List, Dict
from pathlib import Path
from pandas import DataFrame, merge
from numpy import ndarray, array_equal
from torch import (
    uint8,
    float32,
    Tensor,
    save as pyt_save,
    load as pyt_load,
    from_numpy as pyt_from_numpy
)
from torch.utils.data import Dataset
from scipy.io import mmread, mmwrite
from .embedding import Embedder
import gzip
from warnings import warn
from tqdm import tqdm


class CnvDataset(Dataset):
    """
    Dataset class for DNA, ATAC and CNV embeddings.
    Manages a dataset defined by a pandas.DataFrame with columns
    * barcode
    * gene_id
    * expression_count
    * classification
    Single data points are stored as files in the background.
    This class wraps all the I/O for accessing the data.
    Furthermore, it can also be used to compute the embeddings for a new dataset.
    In this case the parameters fasta_path, gtf_path, atac_path and cnv_path are
    required keyword arguments.

    Some useful attributes of this class (each with according init parameter):
    * dtype: define the torch.dtype or numpy.dtype of returned values
    * file_format: define how to store embedding files (options: 'mtx', 'pt')
    * return_numpy: return numpy.ndarray instead of torch.Tensor
    * target_type: return either a regression or classification target
    """

    def __init__(self, root, data_df: DataFrame, force_recompute=False,
                 embedding_mode='single_gene_barcode', file_format='pt',
                 use_gzip=False, verbose=1, dtype=float32, return_numpy=False, 
                 target_type='classification', **kwargs):
        """
        Initialize Dataset from given parameters.

        If no data is found at root, or recompute is forces, this function tries
        to compute embeddings from raw data. In this case use kwargs:
        * fasta_path: str as path to reference genome fasta file
        * gtf_path: str as path to reference genome annotation file
        * atac_path: str as path to peaks file from ATAC-seq
        * cnv_path: str as path to EpiAneufinder results

        Parameters:
        root : str as data root path
        data_df : pandas.DataFrame with columns:
            'barcode', 'gene_id', 'expression_count', 'classification'
            where 'barcode' and 'gene_id' represent the cell barcode and ENSEMBL
            id of the respective data point while 'expression_count' and
            'classification' are the regression and classification targets.
        force_recompute : bool if to force recomputation of embeddings
        embedding_mode : str as mode of embedding computation. Currently only
            'single_gene_barcode' supported.
        file_format : str one of ('mtx', 'pt') specifying which format to use
            (.mtx gives smaller files, while .pt allows 1000x faster loading)
        use_gzip : bool weather to use gzip compression for I/O
        verbose : int of verbosity level (0: silent, 4: max verbosity)
        dtype : torch.dtype of Tensor/array dtype to return.
        return_numpy : bool if to return numpy.ndarray indead of torch.Tensor
        target_type : str one of ('classification', 'regression') to specify
            type of supervised learning target

        TODO:
        * pre-load embeddings from file to buffer I/O time
          file size of one single_gene_barcode matrix: 561463 bytes
        """

        assert not data_df.empty, 'DataFrame data_df is empty!'
        assert embedding_mode == 'single_gene_barcode', \
            'Unsupported embedding mode: {}'.format(embedding_mode)
        assert file_format in ('pt', 'mtx'), 'File format not supported!'
        assert target_type in ('classification', 'regression'),\
            'Unknown target_type: {} != "classification" / "regression"'.format(
                target_type
            )
        
        super().__init__()

        self.root_path = Path(root) / embedding_mode
        self.data_df = data_df
        self.embedding_mode = embedding_mode
        self.file_format = file_format
        self.compress = use_gzip
        self.dtype = dtype
        self.return_numpy=return_numpy
        if target_type == 'classification':
            self.target_type = target_type
        elif target_type == 'regression':
            self.target_type = 'expression_count'
        else:
            raise RuntimeError('Unknown target type: {}'.format(target_type))
        self.embedding_rows = self._subset_embedding_rows(
            dna=kwargs.get('include_dna', True),
            atac=kwargs.get('include_atac', True),
            cnv=kwargs.get('include_cnv', True)
        )

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

        if recompute:
            if verbose > 0:
                print('Recomputing embeddings: ', recompute)

            barcodes_to_genes = {
                barcode: list(set(
                    data_df[data_df['barcode'] == barcode]['gene_id']
                ))
                for barcode in set(data_df['barcode'])
            }

            embedder = Embedder(
                fasta_path=kwargs.get('fasta_path'),
                gtf_path=kwargs.get('gtf_path'),
                atac_path=kwargs.get('atac_path'),
                cnv_path=kwargs.get('cnv_path'),
                barcodes_to_genes=barcodes_to_genes,
                verbose=(verbose > 2)
            )

            path_list = list()
            barcode_list = list()
            gene_id_list = list()
            for barcode, gene_id, embedding in embedder:
                if verbose > 3:
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

            self.data_df = merge(
                self.data_df,
                emb_df,
                how='right',
                on=['barcode', 'gene_id']
            )
        
        else:
            # only create paths for data_df and check if they exist
            self.data_df['embedding_path'] = [
                self.ids_to_emb_path(b, g) for b, g in 
                self.data_df[['barcode', 'gene_id']].itertuples(index=False)
            ]

        # report missing/unused embeddings for both cases
        if verbose > 2:
            print('data_df before missing files drop') # merge_df
            print(self.data_df)

        # print missing
        missing_df = self.data_df[self.data_df.apply(
            lambda x: not x['embedding_path'].exists(),
            axis=1
        )]
        if not missing_df.empty:
            print('No embedding files for {} data points in {}!'.format(
                missing_df.shape[0], self.root_path
            ))
            if verbose > 1:
                print(missing_df['embedding_path'])
        elif missing_df.shape[0] == self.data_df.shape[0]:
            raise RuntimeError('No embedding files found!')
        
        self.data_df = self.data_df.drop(missing_df.index)

        # define class to label matching if we have a classification
        if self.target_type == 'classification':
            self.class_to_label = {
                c: i for i, c in 
                enumerate(self.data_df[self.target_type].unique())
            }
            self.label_to_class = {i: c for c, i in self.class_to_label.items()}

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
                mmwrite(file, embedding, field='integer')
    
    @staticmethod
    def _load_embedding(file_path: Path, **kwargs):
        """
        Load an embedding from file depending on the file format.
        """

        dtype = kwargs.get('dtype', uint8)
        return_numpy = kwargs.get('return_numpy', False)

        file_format = file_path.name.split('.')[-1]
        if file_format == 'gz':
            file_path = gzip.open(file_path)
            file_format = file_path.name.split('.')[-2]

        match file_format:
            case 'pt':
                    if return_numpy:
                        return pyt_load(file_path).to(dtype).numpy()
                    return pyt_load(file_path).to(dtype)
            case 'mtx':
                try:
                    if return_numpy:
                        return mmread(file_path)
                    return pyt_from_numpy(mmread(file_path)).to(dtype)
                except ValueError as e:
                    raise e.add_note('could not read file {}'.format(file_path))
        raise RuntimeError('Unsupported file format: {}'.format(file_format))

    @staticmethod
    def _subset_embedding_rows(dna=True, atac=True, cnv=True):
        """
        Create a list to filter rows of an embedding.
        
        n_rows : int of number of rows of an embedding tensor
        dna : bool if include DNA sequence coding rows, that is rows 0 to 3
        atac : bool if include open chromatin row (4)
        cnv : bool if include CNV encoding rows (5 and 6)
        """

        n_rows = 7
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
    def _get_embedding(data_df, idx: int, rows:Union[List[int], None]=None,
                       **kwargs):
        """
        Loads embeddings from file and filters rows.

        data_df : pandas.DataFrame with column 'embedding_path'
        idx : int of the embedding to load
        rows : None or List[int] of row indices to include from embedding. This
            parameter is not sanity checked. Make sure to create this list with
            the _subset_embedding_rows() function.
        """

        file_path = data_df.iloc[idx]['embedding_path']
        embedding = CnvDataset._load_embedding(file_path, **kwargs)

        if rows is not None:
            return embedding[rows,:]
        return embedding

    def _get_grund_truth(self, idx: int):
        """
        Return the ground truth respective to regression or classification.

        data_df : pandas.DataFrame with columns:
            'barcode', 'gene_id', 'expression_count', 'classification'
            where 'barcode' and 'gene_id' represent the cell barcode and ENSEMBL
            id of the respective data point while 'expression_count' and
            'classification' are the regression and classification targets.
        idx : int of index in data_df
        type : str, one of 'expression_count' or 'classification'

        returns: Tuple[torch.Tensor, torch.Tensor] of the embedding and the target
            value of
        """
        
        if self.target_type == 'classification':
            label = self.class_to_label[self.data_df.iloc[idx][self.target_type]]
            label = Tensor([label]).to(self.dtype)
            if self.return_numpy:
                return label.numpy()
            return label
        else:
            # regression case
            if self.return_numpy:
                return self.data_df.iloc[idx][self.target_type]
            else:
                return Tensor([self.data_df.iloc[idx][self.target_type]])

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx, **kwargs):
        return (
            self._get_embedding(
                self.data_df, idx, dtype=self.dtype,
                return_numpy=self.return_numpy, 
                rows=self.embedding_rows, **kwargs
                ),
            self._get_grund_truth(idx)
        )
        
    def get_with_ids(self, idx, **kwargs):
        return (
                self.data_df.iloc[idx]['barcode'],
                self.data_df.iloc[idx]['gene_id'],
                self._get_embedding(
                self.data_df, idx, dtype=self.dtype,
                return_numpy=self.return_numpy, **kwargs
                ),
                self._get_grund_truth(idx)
            )

    def __repr__(self):
        return '{} with {} datapoints'.format(self.__class__, len(self))

    def __str__(self):
        return '{} with {} datapoints\n{}'.format(
            self.__class__, len(self), self.data_df.head()
        )

    def class_balance(self, counts=True):
        assert self.target_type == 'classification', \
            'Only class balance for classification'
        class_counts = self.data_df[self.target_type].value_counts()
        if counts:
            return class_counts
        else:
            return class_counts / self.data_df.shape[0]

    def convert_file_format(self, old_format: str, new_format: str, 
                            convert=True, check=True, remove_old=False):
        supported_formats = ('mtx', 'pt')
        assert old_format in supported_formats and \
            new_format in supported_formats, 'Unsupported format!'
        assert old_format != new_format, 'Not converting to same format!'

        if convert:
            print('Converting files from .{} to .{}'.format(
                old_format, new_format
            ))
            for i in tqdm(range(len(self))):
                barcode, gene_id, embedding, _ = self.get_with_ids(i)
                path = self.ids_to_emb_path(barcode, gene_id)
                new_path = Path(str(path).replace(old_format, new_format))
                self._save_embedding(embedding, new_path)

        if check:
            old_return_numpy = self.return_numpy
            self.return_numpy = True
            print('Sanity checking new files.')
            if remove_old:
                print('Removing old files after sanity check.')
            new_dataset = CnvDataset(
                root=self.root_path.parent,
                data_df=self.data_df,
                file_format=new_format,
                return_numpy=self.return_numpy
            )
            for i in tqdm(range(len(self))):
                b_old, g_old, e_old, _ = self.get_with_ids(i)
                b_new, g_new, e_new, _ = new_dataset.get_with_ids(i)
                assert b_old == b_new and g_old == g_new, \
                    'ID mismatch {}: {}, {} != {}, {}'.format(
                        i, b_old, g_old, b_new, g_new
                    ) # TODO: check array_equal(Tensor, Tensor)
                assert array_equal(e_old, e_new), \
                    'Array mismatch for {}: {}, {}; {} ?= {}\n{}\n{}'.format(
                        i, b_old, g_old, e_old.dtype, e_new.dtype, e_old, e_new
                    )
        
                if remove_old:
                    old_path = self.ids_to_emb_path(b_old, g_old)
                    if old_path.is_file():
                        old_path.unlink()
            
            print('All files checked.')
            if remove_old:
                print('All old files removed.')
            
            self.return_numpy = old_return_numpy

    def split(self, cell_type_df: DataFrame, best_barcodes=None):
        """
        TODO: split existing data based on cell type, barcode and class
        """
        from sklearn.model_selection import train_test_split

        # TODO: visualiuze distributions before split
        
        if best_barcodes is not None:
            assert len(set(best_barcodes).intersection(set(self.data_df['barcode']))) > 0
            self.data_df = self.data_df[self.data_df['barcode'].isin(best_barcodes)]

        merge_df = merge(
            self.data_df, cell_type_df, on='barcode'
        )

        # create compined class labels with cell type and expression class
        merge_df['label'] = merge_df['classification'] + merge_df['celltype']

        # TODO: problem is that gene lists per barcode are inequal of size
        # -> idea make categories by gene list size and distribute evenly for splits
        # -> not truly random split anymore

        # all_barcodes = merge_df['barcode'].unique()
        training_barcodes, test_barcodes = train_test_split(
            merge_df[['barcode', 'gene_id']],
            test_size=0.2,
            random_state=2,
            stratify=merge_df['label']
        )
        train_barcodes, val_barcodes = train_test_split(
            training_barcodes,
            test_size=10/80,
            random_state=2,
            stratify=merge_df[merge_df.index.isin(training_barcodes.index)]['label']
        )

        # TODO: visualize distribution of splits
        
        return (
            merge_df[merge_df.index.isin(train_barcodes)],
            merge_df[merge_df.index.isin(val_barcodes)],
            merge_df[merge_df.index.isin(test_barcodes)],
        )


class CnvMemoryDataset(CnvDataset):
    """
    Dataset version that lives completely in memory.
    TODO: implement
    """

    def __init__(self, root, data_df, force_recompute=False, 
                 embedding_mode='single_gene_barcode', file_format='pt', 
                 use_gzip=False, verbose=1, dtype=float32, return_numpy=False, 
                 target_type='classification', **kwargs):
        super().__init__(
            root, data_df, force_recompute, embedding_mode, 
            file_format, use_gzip, verbose, dtype, return_numpy, 
            target_type, **kwargs)
        
        # TODO: check memory requirements

        self.data = [d for d in super()]

    def __getitem__(self, idx, **kwargs):
        return super().__getitem__(idx, **kwargs)
