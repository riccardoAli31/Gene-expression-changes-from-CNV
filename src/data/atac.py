import os

@DeprecationWarning
def generate_gene_peak_overlap(file_path: str):
	"""
	Generator for creating peak and gene regions from overlap between 
	 ATAC-seq peaks and gene regions.
	This file is in tsv format with the columns:
	Chromosome<TAB>Start_peak<TAB>End_peak<TAB>Start_gene<TAB>End_gene<TAB>gene_id
	Where gene_id refers to the ENSEMBL id for the respective gene.

	file_path : str of path to overlap file
	"""

	assert os.path.exists(file_path) and os.path.isfile(file_path)

	with open(file_path, "r") as in_file:
		for line in in_file:
			chrom, peak_start, peak_end, gene_start, \
				gene_end, gene_id = line.strip().split('\t')
			peak_start, peak_end = int(peak_start), int(peak_end)
			gene_start, gene_end = int(gene_start), int(gene_end)

			yield chrom, peak_start, peak_end, gene_start, gene_end, gene_id


