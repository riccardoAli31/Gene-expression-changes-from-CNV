#!/bin/python

"""
Script for creating embeddings from DNA regions.
See https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x/figures/5 
"""

from numpy import ndarray
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Set
import os
from Bio import SeqIO
import gzip

# TODO:
#	- how to handle missing CNV overlaps
#	- read reference genome from .fasta.tar.gz using import tarfile? 
#	- create CNV rows per barcode
#	- filter by genes with high expression change

# TODO: implement two different modes:
# - one returning multiple genes concatenated per cell
# - another returning multiple genes as channels for each gene

# Possible bugs:
# - close to chromosomal end
# - capital and small letters could be in ref genome -> use only capital or small letter


autosomes = list(map(str, range(1,22)))
allosomes = ['X', 'Y']
standard_chromosomes = list(map(str, range(1,22)))
standard_chromosomes.extend(allosomes)

def relative_idx(idx: int, interval: Tuple[int,int],
				 correct: int = 0, clip_start=True,
				 clip_end=True):
	"""
	Calculates relative index inside an interval from an index based outside the interval.
	E.g. index=42, interval=(13, 50) -> relative index is 29
	This function is intended to be used on chromosomal indices and intervals.

	idx : int of index based outside the interval
	interval : Tuple[int,int] of interval start and end coordinates
	correct : int of index shift correction. I.e. from 0-based to 1-based use +1.
	clip_* : bool wether idx should be interval start/end, if it exceeds interval start/end.
		Otherwise an AssertionError will be raised.
	"""

	start, end = interval
	assert start < end

	if clip_start and idx < start:
		idx = start
	
	if clip_end and end < idx:
		idx = end
	
	assert start <= idx <= end
	
	return idx - start + correct


def interval_overlap(interval_1: Tuple[int,int], 
		interval_2: Tuple[int,int]) -> Tuple[int,int]:
	"""
	Calculates the start and end position of an overlap 
	between two intervals.

	"""
	start_1, end_1 = interval_1
	start_2, end_2 = interval_2
	
	assert start_1 < end_1 and start_2 < end_2

	if end_1 < start_2 or end_2 < start_1:
		return None
	
	start = start_1 if start_2 <= start_1 else start_2
	end = end_1 if end_1 <= end_2 else end_2

	return (start, end)


def extract_cnv_overlaps(cnv_df: pd.DataFrame, barcode: str,
		region: Tuple[str,int,int]) -> List[Tuple[Tuple[int, int],int]]:
	"""
	Extracts all overlaps between rows in the copy number status 
	data frame and the genomic region of interest (e.g. embedding)

	"""

	r_chrom, r_start, r_end = region
	overlap_rows = cnv_df[['seq', 'start', 'end', barcode]]\
		[(cnv_df['seq'] == r_chrom) & (
			((cnv_df['start'] <  r_end) & (r_end < cnv_df['end'])) |
			((cnv_df['start'] < r_start) & (r_start < cnv_df['end'])) | 
			((r_start < cnv_df['start']) & (cnv_df['end'] < r_end)))]

	# TODO: how to handle missing values / CNV overlaps?
	if overlap_rows.empty:
		return [((),)]
	
	return [
		(interval_overlap((r_start, r_end), (c_start, c_end)), c_status)
		for _, (_, c_start, c_end, c_status) in overlap_rows.iterrows()
	]


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
							  path_to_gft='data/Homo_sapiens.GRCh38.113.gtf.gz',
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

	with gzip.open(path_to_gft, 'r') as gtf_file:
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


def encode_dna_seq(dna: str, alphabet="ACGT", **kwargs) -> ndarray:
	"""
	Produces an encoding for DNA sequences, based on the IUPAC codes.
	See: https://genome.ucsc.edu/goldenPath/help/iupac.html
	This encoding only represents standard amino acids by one row each
	 in a (4 x sequence_length) matrix. The rows are ordered ACGT.

	dna_string : str of DNA sequence with posibly N-padding
	alphabet : str of unique bases to encode. If you want to encode the 
			N padding as a separate row, please adapt this here.  
	"""
	
	dna = dna.upper()

	nucleotide_encoding = {
		"A": np.array([1, 0, 0, 0], dtype='u1', ndmin=2).T,
		"C": np.array([0, 1, 0, 0], dtype='u1', ndmin=2).T,
		"G": np.array([0, 0, 1, 0], dtype='u1', ndmin=2).T,
		"T": np.array([0, 0, 0, 1], dtype='u1', ndmin=2).T,
		# "U": np.array([0, 0, 0, 1], dtype='u1', ndmin=2).T,
		"R": np.array([1, 0, 1, 0], dtype='u1', ndmin=2).T,
		"Y": np.array([0, 1, 0, 1], dtype='u1', ndmin=2).T,
		"M": np.array([1, 1, 0, 0], dtype='u1', ndmin=2).T,
		"K": np.array([0, 0, 1, 1], dtype='u1', ndmin=2).T,
		"S": np.array([0, 1, 1, 0], dtype='u1', ndmin=2).T,
		"W": np.array([1, 0, 0, 1], dtype='u1', ndmin=2).T,
		"B": np.array([0, 1, 1, 1], dtype='u1', ndmin=2).T,
		"D": np.array([1, 0, 1, 1], dtype='u1', ndmin=2).T,
		"H": np.array([1, 1, 0, 1], dtype='u1', ndmin=2).T,
		"V": np.array([1, 1, 1, 0], dtype='u1', ndmin=2).T,
		"N": np.array([1, 1, 1, 1], dtype='u1', ndmin=2).T,
		# "X": np.array([1, 1, 1, 1], dtype='u1', ndmin=2).T,
		# "-": np.array([0, 0, 0, 0], dtype='u1', ndmin=2).T,
	}

	# encoded_dna = np.zeros((len(alphabet), len(dna)), dtype='u1')
	encoded_dna = [nucleotide_encoding[aa] for aa in dna]
	return np.hstack(encoded_dna)


def encode_open_chromatin(embedding_interval: Tuple[int, int], 
						  peak_intervals: List[Tuple[int, int]],
						  **kwargs) -> ndarray:
	"""
	Creates the open chromatin embedding part, that observes ones all
	genmoic positions that are in open chromatin and 0 for closed.

	embedding_interval: Tuple[int,int] of genomic start and end for the 
		region to create an embedding for
	peak_intervals : List[Tuple[int,int]] of start and end position for a
		list of peaks from ATAC-seq  
	"""

	emb_start, emb_end = embedding_interval
	assert emb_start < emb_end
	atac_embedding = np.zeros((1, emb_end - emb_start), dtype='u1')
	for (peak_start, peak_end) in peak_intervals:
		if emb_end <= peak_start:
			continue
		if emb_end < peak_end:
			peak_end = emb_end

		# sanity check there is an overlap between gene and peak
		assert peak_start < emb_end and emb_start < peak_end

		peak_start = emb_start if peak_start < emb_start else peak_start
		peak_end = peak_end if peak_end < emb_end else emb_end

		# calculate relative start stop on embedding
		atac_embedding[peak_start - emb_start:peak_end - emb_start] = 1
		
	return atac_embedding


def encode_cnv_status(embedding_interval: Tuple[int, int],
					  cnv_interval_status: List[Tuple[Tuple[int, int],int]],
					  **kwargs):
	"""
	Creates two 0/1 vector encoding both copy number gains and losses.
	Thus, the gain vector will be 0 at all normal or loss positions and
	 vice versa.
	Both vectors are stacked to a 2d array.

	cnv_status : List[int] of copy number status, as follows:
		0: 'loss', 1: 'normal', 2: 'gain'
	"""

	emb_start, emb_end = embedding_interval
	emb_length = emb_end - emb_start

	# TODO: how to handle missing values / CNV overlaps?
	if cnv_interval_status == [((),)]:
		return np.zeros((2, emb_length))

	cnv_loss = np.zeros(emb_length)
	cnv_gain = np.zeros(emb_length)

	for (start, end), status in cnv_interval_status:
		start = relative_idx(start, embedding_interval)
		end = relative_idx(end, embedding_interval)
		if status == 0:
			cnv_loss[start:end] = 1
		elif status == 2:
			cnv_gain[start:end] = 1

	return np.vstack([cnv_loss, cnv_gain])


@PendingDeprecationWarning
def encode_cds_structure(cds_intervals: List[Tuple[int, int]], seq_len: int, **kwargs) -> ndarray:
	# TODO
	cds_structure = np.zeros(seq_len, dtype='u1')
	for start, stop in cds_intervals:
		cds_structure[start:stop] = 1

	return cds_structure


@PendingDeprecationWarning
def encode_reading_frame(CDS_start_stop: List[Tuple[int, int]],seq_len: int, **kwargs):
	# TODO compute reading frame from CDS structure
	codon_size = 3
	reading_frame = np.zeros(seq_len, dtype='u1')
	overhead = 0
	for start, stop in CDS_start_stop:
		reading_frame[list(start + (codon_size - overhead), stop, codon_size)] = 1
		overhead = (stop - start + overhead) % 3

	return reading_frame


def embed_dna(gene_regions: List[Tuple[str,int,int]], fasta_path: str,
		embedding_window: Tuple[int,int], pad_dna=True, verbose=False):
	# generate DNA embeddings
	assert os.path.isfile(fasta_path), "FASTA file not found: {}".format(fasta_path)
	fasta_generator = generate_fasta_entries(fasta_path)
	emb_upstream, emb_downstream = embedding_window

	chrom_id, chrom_seq = next(fasta_generator)
	print("chrom_id:", chrom_id) if verbose else None
	for chrom, gene_start, gene_end in gene_regions:
		# assert that the correct chromosome fasta entry is loaded
		while chrom_id != chrom:
			chrom_id, chrom_seq = next(fasta_generator)
			print("chrom_id:", chrom_id) if verbose else None
		assert chrom_id == chrom, "Chromosome mismatch {} != {}".format(chrom_id, chrom)

		emb_start, emb_end = gene_start - emb_upstream, gene_start + emb_downstream

		# get DNA subsequence and apply padding
		dna_seq = chrom_seq[emb_start:emb_end]
		if pad_dna:
			dna_seq = dna_padding(dna_seq, relative_idx(gene_end, (emb_start, emb_end)))

		yield (chrom, emb_start, emb_end, encode_dna_seq(dna_seq))
	

def embed_atac(atac_df: pd.DataFrame, regions, embedding_window: Tuple[int,int]):
	emb_upstream, emb_downstream = embedding_window
	for chrom, gene_start, gene_end in regions:
		emb_start, emb_end = gene_start - emb_upstream, gene_start + emb_downstream
		gene_df = atac_df[
			(atac_df['Chromosome'] == chrom) & 
			(atac_df['Start_gene'] <= gene_start) &
			(atac_df['End_gene'] >= gene_end)
		]

		# create list of peaks for all peaks in region
		peaks = [
			(int(s), int(e)) for s, e in 
			gene_df[['Start_peak', 'End_peak']].to_numpy()
		]

		yield (
			chrom, emb_start, emb_end,
			encode_open_chromatin(embedding_interval=(emb_start, emb_end),
				peak_intervals=peaks)
		)



def embed_cnv(cnv_path: str, regions, mode='gene_concat'):
	# load copy number data
	cnv_df = pd.read_csv(cnv_path, sep=' ')
	cnv_df = cnv_df.sort_values(by=['seq', 'start', 'end'])
	cnv_df['seq'] = pd.Series(map(lambda x: x.replace('chr', ''), cnv_df['seq']))

	# TODO switch for loops depending on mode
	for barcode in cnv_df.columns[4:]:
		for chrom, start, end in regions:
			yield (
				barcode,
				encode_cnv_status(
					(start, end),
					extract_cnv_overlaps(cnv_df, barcode, (chrom, start, end))
				)
			)



def main(fasta_path, atac_path):

	# generate different parts of embedding
	# gtf_generator = generate_genomic_regions(gtf_path)
	fasta_generator = generate_fasta_entries(fasta_path)

	# load open chromatin peaks
	atac_df = pd.read_csv(atac_path, sep='\t')
	# sort autosomes on integer index
	atac_df_auto = atac_df[atac_df['Chromosome'].isin(autosomes)].copy()
	atac_df_auto['Chromosome'] = atac_df_auto['Chromosome'].astype(np.uint8)
	atac_df_auto = atac_df_auto.sort_values(by=['Chromosome', 'Start_gene', 'End_gene'])
	# sort allosomes separately
	atac_df_allo = atac_df[atac_df['Chromosome'].isin(allosomes)].copy()
	atac_df_allo = atac_df_allo.sort_values(by=['Chromosome', 'Start_gene', 'End_gene'])
	atac_df_auto['Chromosome'] = atac_df_auto['Chromosome'].astype(str)
	# concat sorted dataframes
	atac_df = pd.concat([atac_df_auto, atac_df_allo])
	uniq_gene_ids = list(atac_df['gene_id'].unique())


	


def embed(fasta_path, atac_path, cnv_path, mode='gene_concat',
		  gtf_path=None, gene_set: Union[Set[str], None]=None,
		  embed_funcs=[encode_dna_seq, encode_open_chromatin],
		  n_upstream=2000, n_downstream=8000, pad_dna=True):
	"""
	Main wrapper function. Generates embeddings from following files:
	* gtf genome annotation
	* fasta genome sequence
	* overlap data of ATAC-seq peaks and genes

	gtf_path : 
	fasta_path :
	atac_path : 
	cnv_path :  
	mode : str one of ['gene_concat', 'barcode_channel'] to specify how
		to combine embeddings
	"""

	verbose = False
	sanity_check_regions = False
	
	# generate different parts of embedding
	gtf_generator = generate_genomic_regions(gtf_path)
	fasta_generator = generate_fasta_entries(fasta_path)

	# load open chromatin peaks
	atac_df = pd.read_csv(atac_path, sep='\t')
	# sort autosomes on integer index
	atac_df_auto = atac_df[atac_df['Chromosome'].isin(autosomes)].copy()
	atac_df_auto['Chromosome'] = atac_df_auto['Chromosome'].astype(np.uint8)
	atac_df_auto = atac_df_auto.sort_values(by=['Chromosome', 'Start_gene', 'End_gene'])
	# sort allosomes separately
	atac_df_allo = atac_df[atac_df['Chromosome'].isin(allosomes)].copy()
	atac_df_allo = atac_df_allo.sort_values(by=['Chromosome', 'Start_gene', 'End_gene'])
	atac_df_auto['Chromosome'] = atac_df_auto['Chromosome'].astype(str)
	# concat sorted dataframes
	atac_df = pd.concat([atac_df_auto, atac_df_allo])
	uniq_gene_ids = list(atac_df['gene_id'].unique())

	# apply subsetting by gene_id
	if gene_set is not None:
		uniq_gene_ids = set(uniq_gene_ids).intersection(gene_set)

	# dict to save barcode independent embedding data
	gene_embeddings = {}
	gene_positions = {}
	
	chrom_id, chrom_seq = next(fasta_generator)
	print("chrom_id:", chrom_id) if verbose else None
	for gene_id in uniq_gene_ids:

		gene_df = atac_df[atac_df['gene_id'] == gene_id]
		assert gene_df[['Chromosome', 'Start_gene', 'End_gene', 'gene_id']]\
			.drop_duplicates().shape == (1, 4), "Ambiguous gene region for {}".format(gene_id)
		chrom = str(*gene_df['Chromosome'].unique())
		gene_start = int(*gene_df['Start_gene'].unique())
		gene_end = int(*gene_df['End_gene'].unique())
		print(gene_id, 'on', chrom, ':', gene_start, '-', gene_end) if verbose else None

		# assert that the correct chromosome fasta entry is loaded
		while chrom_id != chrom:
			chrom_id, chrom_seq = next(fasta_generator)
			print("chrom_id:", chrom_id) if verbose else None
		assert chrom_id == chrom, "Chromosome mismatch {} != {}".format(chrom_id, chrom)

		# sanity check genomic regions
		if sanity_check_regions:
			gtf_chrom, entry_type, gtf_start, gtf_end, strand, \
				gtf_gene_id = next(gtf_generator)
			while gtf_chrom == chrom and gtf_gene_id != gene_id:
				gtf_chrom, entry_type, gtf_start, gtf_end, strand, \
					gtf_gene_id = next(gtf_generator)
			if entry_type == 'gene':
				assert gtf_start == gene_start and gtf_end == gene_end, \
					"Gene coordinates mismatch for {}:\n{}:{}-{} != {}:{}-{}"\
					.format(gene_id, chrom, gene_start, gene_end, gtf_chrom, gtf_start, gtf_end)

		# compute embedding coordinates
		emb_start = gene_start - n_upstream
		emb_end = gene_start + n_downstream

		peaks = [(int(s), int(e))for s, e in gene_df[['Start_peak', 'End_peak']].to_numpy()]

		# get DNA subsequence and apply padding
		dna_seq = chrom_seq[emb_start:emb_end]
		if pad_dna:
			dna_seq = dna_padding(dna_seq, relative_idx(gene_end, (emb_start, emb_end)))

		# create embedding parts
		embedding = [
			f(
				dna=dna_seq,
				embedding_interval=(emb_start, emb_end),
				peak_intervals=peaks,
				# TODO: CDS data
			) for f in embed_funcs
		]
		
		# combine embedding to one numpy array
		embedding = np.vstack(embedding)

		# save reference genome information in memory
		gene_embeddings[gene_id] = embedding
		gene_positions[gene_id] = (chrom, gene_start, gene_end, emb_start, emb_end)

	# load copy number data
	cnv_df = pd.read_csv(cnv_path, sep=' ')
	cnv_df = cnv_df.sort_values(by=['seq', 'start', 'end'])
	cnv_df['seq'] = pd.Series(map(lambda x: x.replace('chr', ''), cnv_df['seq']))

	# combine embeddings based on mode
	match mode:
		case 'gene_concat':
			#	- for loop over all genes and create embeddings for barcode independet data
			#		(save this in memory if possible)
			#	- for loop over all barcodes
			#	- return for each barcode a Tensor of shape (n_data_rows, sequence_length * n_genes)
			for barcode in cnv_df.columns[4:]:
				# use list comprehension to create full embedding to return
				# TODO be mindful of your memory, depending on the number 
				#  of uniq genes this may become very large 
				yield np.hstack([
					np.vstack(
						[gene_embeddings[gene_id],
						encode_cnv_status(
							(emb_start, emb_end),
							extract_cnv_overlaps(cnv_df, barcode, (chrom, emb_start, emb_end))
						)]
					) for gene_id, (chrom, _, _, emb_start, emb_end) in gene_positions.items()
				])

		case 'barcode_channel':
			# TODO
			#	- for loop over all genes
			#	- return for each gene a Tensor of shape (n_data_rows, sequence_length, n_barcodes)
			pass
		

