from src.data import generate_genomic_regions

def test_generate_genomic_regions():
    first_seven_genes = [
        'ENSG00000142611', 'ENSG00000284616', 'ENSG00000157911',
        'ENSG00000260972', 'ENSG00000224340', 'ENSG00000229280',
        'ENSG00000142655'
    ]
    gtf_generator = generate_genomic_regions(first_seven_genes)
    extracted_genes = [next(gtf_generator)[-1] for _ in range(7)]

    assert all(first_seven_genes == extracted_genes)

    # zcat data/Homo_sapiens.GRCh38.113.gtf.gz | grep -E '\Wgene\W' | wc -l
    n_gene_entries = 78932
    i = 7
    for _ in gtf_generator:
        i += 1
    assert n_gene_entries == i
