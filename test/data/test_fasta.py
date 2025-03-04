from src.data import dna_padding, generate_fasta_entries

def test_dna_padding():
	dna = 'CTCTAGCTAGCTAGCTATCG'
	assert dna_padding(dna, 15) == 'CTCTAGCTAGCTAGCNNNNN'


def test_generate_fasta_entries():
    standard_chroms = [
        '1', '2', '3', '4', '5', '6', '7', 
        '8', '9', '10', '11', '12', '13', 
        '14', '15', '16', '17', '18', '19',
        '20', '21', 'X', 'Y'
    ]
    base_counts = {}
    for chrom, seq in generate_fasta_entries():
        assert chrom in standard_chroms
        assert len(seq) > 0
        seq_counts = {base: seq.count(base) for base in set(seq)}
        base_counts = {
            base: base_counts[base] + count
            if base in base_counts else count 
            for base, count in seq_counts.items()
        }

    print(base_counts)

    all_chroms = [
        chrom for chrom, _ in generate_fasta_entries(only_standard_chrom=False)
    ]
    assert len(standard_chroms) <= len(all_chroms)
