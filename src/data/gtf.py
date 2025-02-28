import gzip
from typing import List

def extract_gtf_annotation(gtf_line: str, field):
		"""
		Helper function to extact a field from gft annotation column.
		
		"""
		# TODO: use field_name variable to generalize this function
		# gtf_annotation.split(';')[0].split('\"')[-2]
		if gtf_line.startswith('#'):
			return ''
		gtf_annotation = gtf_line.split('\t')[-1]
		gtf_annotation = {splt.split('\"')[0].strip(): splt.split('\"')[1].strip() for splt in gtf_annotation.split(';')}
		
		return gtf_annotation[field]


def generate_genomic_regions(gene_ids: List[str],
							  path_to_gft='data/reference/Homo_sapiens.GRCh38.113.gtf.gz',
							  entry_types={'gene'}):
	"""
	Generates genomic regions for genes of interest.

	gene_ids : List of str containting ENSEMBL gene identifiers (e.g. ENSG00000142611)
				This list should be sorted by genomic position.
	path_to_gft : str of the path to .gft file
	entry_types : Set[str] of gft entry types to extract
	"""

	# sample gft line
	# 1	ensembl_havana	gene	3069168	3438621	.	+	.	gene_id "ENSG00000142611"; gene_version "17"; gene_name "PRDM16"; gene_source "ensembl_havana"; gene_biotype "protein_coding";
	
	# TODO remove gene ids and filter after extraction on entry type
	gene_id_iterator = iter(gene_ids)

	with gzip.open(path_to_gft, 'rt') as gtf_file:
		gene_id = next(gene_id_iterator)
		gene_id_seen = False
		for line in gtf_file:
			line_gene_id = extract_gtf_annotation(line, 'gene_id')
			if line_gene_id == gene_id:
				gene_id_seen = True
				line_split = line.strip().split('\t')
				if line_split[2] in entry_types:
					# TODO assert gene_biotype "protein_coding"?
					chrom, _, entry_type, start, end, _, strand = line_split[:6]
					yield (str(chrom), entry_type, int(start), int(end), strand, gene_id)

			elif gene_id_seen:
					gene_id = next(gene_id_iterator)
					if gene_id == gene_id_iterator:
						# TODO hanle case when next gene id follows directly after
						pass
					else:
						gene_id_seen = False

