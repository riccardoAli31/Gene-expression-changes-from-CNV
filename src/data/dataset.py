import torch
from src.dataloader.embedding import embed
import os
import pandas as pd

class CopyNumerDNADataset(torch.utils.data.DataSet):
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

        self.root_path = root

        if not os.path.isdir(root) or force_recompute:
            fasta_path = kwargs.get('fasta_path')
            atac_path = kwargs.get('atac_path')
            cnv_path = kwargs.get('cnv_path')
            if not os.path.isdir(root):
                os.mkdir(root)

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
                                [root, embedding_mode, barcode, gene_id + '.pt']
                            )
                        )
                    case 'gene_concat':
                        torch.save(
                            torch.from_numpy(embedding), 
                            os.path.join(
                                [root, embedding_mode, barcode, + '.pt']
                            )
                        )
                    case 'barcode_channel':
                        torch.save(
                            torch.from_numpy(embedding), 
                            os.path.join(
                                [root, embedding_mode, gene_id + '.pt']
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
                            os.path.join([root, embedding_mode, cell + '.pt'])
                            for cell in barcode_ids
                        ]
                    }
                )
            case 'barcode_channel':
                files_df = pd.DataFrame(
                    {
                        'file_path':
                        [
                            os.path.join([root, embedding_mode, gene + '.pt'])
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
                                [root, embedding_mode, barcode, gene + '.pt']
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
    def _get_embedding(file_path_df, idx, dna=True, atac=True, cnv=True):
        """
        Loads embeddings from file and filters rows.

        file_path_df : pandas.DataFrame with column 'file_path'
        ids : int of the embedding to load
        dna : bool if include DNA sequence coding rows, that is rows 0 to 3
        atac : bool if include open chromatin row (4)
        cnv : bool if include CNV encoding rows (5 and 6)
        """

        # load embedding
        embedding = torch.load(file_path_df.loc[idx]['file_path'])

        # filter rows to select
        rows = []
        if dna:
            rows.extend([0, 1, 2, 3])
        if atac:
            rows.extend([4])
        if cnv:
            rows.extend([5, 6])
        
        # sanity check rows are valid
        # assert all(map(lambda x: x < embedding.shape[0], rows))

        return embedding[rows,:]

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
                
