from Bio import SeqIO
from .genome import standard_chromosomes

def generate_fasta_entries(fasta_path='data/GRCh38.d1.vd1.fa',
						   verbose=False, only_standard_chrom=True):
	"""
	Generates for chromosome fasta entries in the order saved in file.
	Assert that the chromosomes are saved in increasing order, 
	 starting with Autosomes.
	
	Download the GRCh38.d1.vd1.fa.tar.gz refence genome from the [GDC](https://api.gdc.cancer.gov/data/254f697d-310d-4d7d-a27b-27fbf767a834)
	The fasta reference used in this project lists chromosomes like this:
	>chr1  AC:CM000663.2  gi:568336023  LN:248956422  rl:Chromosome  M5:6aef897c3d6ff0c78aff06ac189178dd  AS:GRCh38
	Thus we need to extract the chromosome name from this string
	"""
	
	for entry in SeqIO.parse(fasta_path, 'fasta'):
		seq_id = entry.id.split(' ')[0].replace('chr', '')
		if only_standard_chrom and seq_id not in standard_chromosomes:
			continue
		print("Reading Chromosome:", seq_id) if verbose else None
		seq = str(entry.seq)
		yield seq_id, seq


def dna_padding(dna: str, rel_gene_end: int, pad_char: str='N'):
	"""
	Replaces DNA sequence downstream of a (relative) gene end position
	with padding character.

	dna : str of DNA sequence
	rel_gene_end : int of gene end position on the DNA sequence
	pad_char : str (default 'N') of padding character (length 1 required)
	"""

	assert len(pad_char) == 1
	return dna[:rel_gene_end] + pad_char * (len(dna) - rel_gene_end)

