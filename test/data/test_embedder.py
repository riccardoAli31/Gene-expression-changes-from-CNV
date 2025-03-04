#!/usr/bin/python

from pathlib import Path
from src.data.embedding import Embedder
from typing import Tuple

git_root=Path('.')
gtf_path=git_root / 'data/gene_positions_and_overlaps/gene_positions.csv'
fasta_path=git_root / 'data/reference/GRCh38.d1.vd1.fa'
overlap_path=git_root / 'data/gene_positions_and_overlaps/overlaps_batch1.tsv'
epianeu_path=git_root / 'out/epiAneufinder/epiAneuFinder_results.tsv'
assert all(map(lambda p:p.exists(),
            [gtf_path, fasta_path, overlap_path, epianeu_path]
))


def test_embedder():

    def embedder_test(expected_n_emb: int, expected_emb_shape: Tuple,
                      expected_barcodes=None, expected_genes=None, **kwargs
                      ):
        # embedder creation designed to fail on improper kwards
        embedder = Embedder(
            fasta_path=kwargs.get('fasta_path', fasta_path),
            gtf_path=kwargs.get('gtf_path', gtf_path),
            atac_path=kwargs.get('overlap_path', overlap_path),
            cnv_path=kwargs.get('epianeu_path', epianeu_path),
            barcode_set=kwargs.get('barcode_set', None),
            gene_set=kwargs.get('gene_set', None),
            barcodes_to_genes=kwargs.get('barcodes_to_genes', None),
            verbose=kwargs.get('verbose', False)
        )
        i = 0
        for barc, gid, emb in embedder:
            if expected_barcodes is not None:
                assert barc == expected_barcodes[i], \
                    'Barcode mismatch for {}:{}\nexpec.: {}\nactual: {}'.format(
                        i, gid, expected_barcodes[i], barc
                    )
            if expected_genes is not None:
                assert gid == expected_genes[i], \
                    'GeneID mismatch for {}:{}\nexpec.: {}\nactual: {}'.format(
                        i, barc, expected_genes[i], gid
                    )
            
            assert emb.shape == expected_emb_shape, \
				"Wrong shape: {} != {}".format(emb.shape, expected_emb_shape)
            i += 1
            
        assert i == expected_n_emb, 'Wrong number of embeddings {}!={}'.format(
            expected_n_emb, i
        )
    
    # base test data
    test_genes = [
        'ENSG00000154511',
        'ENSG00000269113',
        'ENSG00000225555',
        'ENSG00000188158'
    ]
    test_barcodes = [
        'AAACCAACATGTCAGC-1',
        'CCCTGTTAGCACGTTG-1',
        'TTGTTTGGTTAATGCG-1'
    ]
    test_barcodes_to_genes = {
        'AAACCAACATGTCAGC-1': ['ENSG00000154511'],
        'CCCTGTTAGCACGTTG-1': ['ENSG00000269113', 'ENSG00000225555'],
        'TTGTTTGGTTAATGCG-1': ['ENSG00000269113', 'ENSG00000154511', 'ENSG00000188158'],
    }

    # single_gene_barcode for single barcode and multiple genes
    embedder_test(
        expected_n_emb=len(test_genes) - 1,
        expected_emb_shape=(7, 10_000),
        expected_genes=test_genes[:-1],
        expected_barcodes=[test_barcodes[0]] * (len(test_genes) - 1),
        barcode_set={test_barcodes[0]},
        gene_set=set(test_genes[:-1]),
        )
    
    # single_gene_barcode for multiple barcodes and single gene
    embedder_test(
        expected_n_emb=len(test_barcodes),
        expected_emb_shape=(7, 10_000),
        expected_genes=[test_genes[0]] * len(test_barcodes),
        expected_barcodes=test_barcodes,
        barcode_set=set(test_barcodes),
        gene_set={test_genes[0]},
        )
    
    # single_gene_barcode test case
    embedder_test(
        expected_n_emb=len(test_genes[:-1]) * len(test_barcodes),
        expected_emb_shape=(7, 10_000),
        expected_genes=[g for gl in test_genes[:-1] for g in [gl] * len(test_barcodes)],
        expected_barcodes=test_barcodes * len(test_genes[:-1]),
        barcode_set=set(test_barcodes),
        gene_set=set(test_genes[:-1]),
        )

    # custom barcode to genes pairing in single_gene_barcode mode
    embedder_test(
        expected_n_emb=5,
        expected_emb_shape=(7, 10_000),
        expected_genes=[
            'ENSG00000154511', 'ENSG00000154511', 'ENSG00000269113', 
            'ENSG00000269113', 'ENSG00000225555'
        ],
        expected_barcodes=[
            'AAACCAACATGTCAGC-1', 'TTGTTTGGTTAATGCG-1', 'CCCTGTTAGCACGTTG-1',
            'TTGTTTGGTTAATGCG-1', 'CCCTGTTAGCACGTTG-1'
            
        ],
        barcode_set=set(test_barcodes),
        gene_set=set(test_genes[:-1]),
        barcodes_to_genes=test_barcodes_to_genes
        )
