import torch
from src.dataloader.embedding import embed
import os
import pandas as pd
from typing import Union, Tuple, List

class CopyNumerDNADataset(torch.utils.data.Dataset):
    """
    Dataset class for DNA, ATAC and CNV data.
    """

    def __init__(self, root, barcode_ids, gene_ids, *args,
                 force_recompute=False, embedding_mode='gene_concat',
                 **kwargs):
        """
        Initialization funciton.
        Computes embeddings from raw data, if needed. In this case use kwargs:
        * fasta_path: str as path to reference genome fasta file
        * atac_path: str as path to peaks file from ATAC-seq
        * cnv_path: str as path to EpiAneufinder results
        """
        super().__init__(*args, **kwargs)

        self.root_path = os.path.join(root, embedding_mode)

        if not os.path.isdir(self.root_path) or force_recompute:
            fasta_path = kwargs.get('fasta_path')
            atac_path = kwargs.get('atac_path')
            cnv_path = kwargs.get('cnv_path')
            if not os.path.isdir(self.root_path):
                os.mkdir(self.root_path)

            embedder = embed(
                fasta_path,
                atac_path,
                cnv_path,
                gene_set=gene_ids,
                barcode_set=barcode_ids,
                mode=embedding_mode
            )

            for barcode, gene_id, embedding in embedder:
                match embedding_mode:
                    case 'single_gene_barcode':
                        torch.save(
                            torch.from_numpy(embedding), 
                            os.path.join(
                                [self.root_path, barcode, gene_id + '.pt']
                            )
                        )
                    case 'gene_concat':
                        torch.save(
                            torch.from_numpy(embedding), 
                            os.path.join(
                                [self.root_path, barcode, + '.pt']
                            )
                        )
                    case 'barcode_channel':
                        torch.save(
                            torch.from_numpy(embedding), 
                            os.path.join(
                                [self.root_path, gene_id + '.pt']
                            )
                        )
        
        # create table with files
        files_df = None
        match embedding_mode:
            case 'gene_concat':
                files_df = pd.DataFrame(
                    {
                        'file_path':
                        [
                            os.path.join([self.root_path, cell + '.pt'])
                            for cell in barcode_ids
                        ]
                    }
                )
            case 'barcode_channel':
                files_df = pd.DataFrame(
                    {
                        'file_path':
                        [
                            os.path.join([self.root_path, gene + '.pt'])
                            for gene in gene_ids
                        ]
                    }
                )
            case 'single_gene_barcode':
                files_df = pd.DataFrame(
                    {
                        'file_path':
                        [
                            os.path.join(
                                [self.root_path, barcode, gene + '.pt']
                            ) 
                            for gene in gene_ids
                        ]
                    }
                )
        
        # TODO: 
        # * add labels
        # * pre-load embeddings from file to buffer I/O time
        # file size of one single_gene_barcode matrix: 561463 bytes
        self.files_df = files_df

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
    def _get_embedding(file_path_df, idx, rows=Union[List[int], None]):
        """
        Loads embeddings from file and filters rows.

        file_path_df : pandas.DataFrame with column 'file_path'
        idx : int of the embedding to load
        rows : None or List[int] of row indices to include from embedding. This
            parameter is not sanity checked. Make sure to create this list with
            the _subset_embedding_rows() function.
        """

        embedding = torch.load(file_path_df.loc[idx]['file_path'])

        if rows is not None:
            return embedding[rows,:]
        return embedding

    @staticmethod
    def _get_grund_truth_label(file_path_df, idx):
        # TODO return ground truth gene expression classification for 
        pass

    def __len__(self):
        return self.files_df.shape[0]

    def __getitem__(self, idx, **kwargs):
        return {
            'embedding': self._get_embedding(idx, **kwargs),
            'label': self._get_grund_truth_label(idx)
        }

    def split(self, train, test, val=None):
        """
        Function to create training, validation and test splits.
        """

        pass
                
