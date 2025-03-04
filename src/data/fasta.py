from Bio import SeqIO
from .genome import standard_chromosomes
from typing import Generator, Any, Tuple

def generate_fasta_entries(fasta_path='data/GRCh38.d1.vd1.fa',
						   verbose=False, only_standard_chrom=True
						   ) -> Generator[Tuple[str,str],Any,Any]:
	"""
	Generates for chromosome fasta entries in the order saved in file.
	Assert that the chromosomes are saved in increasing order, 
	 starting with Autosomes.
	
	Download the GRCh38.d1.vd1.fa.tar.gz refence genome from the
	[GDC](https://api.gdc.cancer.gov/data/254f697d-310d-4d7d-a27b-27fbf767a834)
	The fasta reference used in this project lists chromosomes like this:
	'>chr1  AC:<str>  gi:<int>  LN:<int>  rl:Chromosome	M5:<str>  AS:GRCh38'
	Thus we need to extract the chromosome name from this string

	fasta_path : str path to reference genome .fasta file
	verbose : bool wether to print reading progress
	only_standard_chrom : bool wether to restrict on standard chromosomes only
	"""
	
	for entry in SeqIO.parse(fasta_path, 'fasta'):
		seq_id = entry.id.split(' ')[0].replace('chr', '')
		if only_standard_chrom and seq_id not in standard_chromosomes:
			continue
		print("Reading Chromosome:", seq_id) if verbose else None
		seq = str(entry.seq)
		yield seq_id, seq


def dna_padding(dna: str, rel_gene_end: int, pad_char: str='N') -> str:
	"""
	Replaces DNA sequence downstream of a (relative) gene end position
	with padding character.

	dna : str of DNA sequence
	rel_gene_end : int of gene end position on the DNA sequence
	pad_char : str (default 'N') of padding character (length 1 required)
	"""

	assert len(pad_char) == 1
	return dna[:rel_gene_end] + pad_char * (len(dna) - rel_gene_end)

