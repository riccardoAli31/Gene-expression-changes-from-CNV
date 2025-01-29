#!/bin/python

"""
Script for creating embeddings from DNA regions.
See https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x/figures/5 
"""
from numpy import ndarray
import numpy as np
import pandas as pd
from typing import List, Tuple
import os
from Bio import SeqIO

# TODO: function to produce open chromatin vector
#  - load chromosome from generator
#  - get genomic regions from generator
#  - get substring of chromosome based on genomic region
#  - produce one-hot encoding from substring (TODO: create separate funciton to unit test)
#  - padding function

# TODO: implement two different modes:
# - one returning multiple genes concatenated per cell
# - another returning multiple genes as channels for each gene

# Possible bugs:
# - close to chromosomal end
# - capital and small letters could be in ref genome -> use only capital or small letter


def extract_gtf_annotation(gtf_line: str, field):
		"""
		Helper function to extact a field from gft annotation column.
		
		"""
		# TODO: use field_name variable to generalize this function
		# gtf_annotation.split(';')[0].split('\"')[-2]
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
	
	gene_id_iterator = iter(gene_ids)

	with open(path_to_gft, 'r') as gtf_file:
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
					yield (int(chrom), entry_type, int(start), int(end), strand, gene_id)

			elif gene_id_seen:
					gene_id = next(gene_id_iterator)
					if gene_id == gene_id_iterator:
						# TODO hanle case when next gene id follows directly after
						pass
					else:
						gene_id_seen = False


def generate_fasta_entries(fasta_path='data/GRCh38.d1.vd1.fa.tar.gz'):
	"""
	Generates for chromosome fasta entries in the order saved in file.
	Assert that the chromosomes are saved in increasing order, 
	 starting with Autosomes.
	The fasta reference used in this project lists chromosomes like this:
	>chr1  AC:CM000663.2  gi:568336023  LN:248956422  rl:Chromosome  M5:6aef897c3d6ff0c78aff06ac189178dd  AS:GRCh38
	Thus we need to extract the chromosome name from this string
	"""
	
	for entry in SeqIO.parse(fasta_path, 'fasta'):
		seq_id = entry.id.split(' ')[0].replace('chr', '')
		print("seq_id:", seq_id)
		seq = str(entry.seq)
		print("seq[:10]", seq[:10])
		yield seq_id, seq


def generate_gene_peak_overlap(file_path: str):
	"""
	Generator for creating peak and gene regions from overlap lap.
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


def encode_one_hot_dna(dna: str, alphabet="ACGT", **kwargs) -> ndarray:
	"""
	Function to produce one-hot encodings for DNA

	dna_string : str of DNA sequence with posibly N-padding
	alphabet : str of unique bases to encode. If you want to encode the 
			N padding as a separate row, please adapt this here.  
	"""
	
	dna = dna.upper()
	ignore_N = 'N' not in set(alphabet)

	one_hot_dna = np.zeros((len(alphabet), len(dna)), dtype='u1')
	for i, aa in enumerate(dna):
		if ignore_N and aa == 'N':
			continue
		one_hot_dna[alphabet.index(aa), i] = 1

	return one_hot_dna


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
	atac_embedding = np.zeros(emb_end - emb_start, dtype='u1')
	for (peak_start, peak_end) in peak_intervals:
		if emb_end < peak_start:
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


def encode_CNV_regions():
	# TODO
	pass


def encode_CDS_structure(CDS_start_stop: List[Tuple[int, int]], seq_len: int, **kwargs) -> ndarray:
	# TODO
	CDS_structure = np.zeros(seq_len, dtype='u1')
	for start, stop in CDS_start_stop:
		CDS_structure[start:stop] = 1

	return CDS_structure


def encode_reading_frame(CDS_start_stop: List[Tuple[int, int]],seq_len: int, **kwargs):
	# TODO compute reading frame from CDS structure
	codon_size = 3
	reading_frame = np.zeros(seq_len, dtype='u1')
	overhead = 0
	for start, stop in CDS_start_stop:
		reading_frame[list(start + (codon_size - overhead), stop, codon_size)] = 1
		overhead = (stop - start + overhead) % 3

	return reading_frame


def embed(gtf_path, fasta_path, overlap_path, mode='gene_concat',
		  parts=[encode_one_hot_dna, encode_open_chromatin],
		  up_stream_window=2000, down_stream_window=8000):
	"""
	Main wrapper function. Generates embeddings from following files:
	* gtf genome annotation
	* fasta genome sequence
	* overlap data of ATAC-seq peaks and genes

	mode : str one of ['gene_concat', 'sample_channel'] to specify how
		to combine embeddings
	"""

	sanity_check_regions = False
	
	# generate different parts of embedding
	gtf_generator = generate_genomic_regions(gtf_path)
	fasta_generator = generate_fasta_entries(fasta_path)
	# atac_generator = generate_gene_peak_overlap(overlap_path)

	atac_df = pd.read_csv(overlap_path, sep='\t')
	atac_df = atac_df.sort_values(by=['Chromosome', 'Start_gene', 'End_gene'])
	uniq_gene_ids = list(atac_df['gene_id'].unique())

	chrom_id, chrom_seq = next(fasta_generator)
	for gene_id in uniq_gene_ids:

		gene_df = atac_df[atac_df['gene_id'] == gene_id]
		assert gene_df[['Chromosome', 'Start_gene', 'End_gene', 'gene_id']]\
			.drop_duplicates().shape == (1, 4), "Ambiguous gene region for {}".format(gene_id)
		chrom = str(*gene_df['Chromosome'].unique())
		gene_start = int(*gene_df['Start_gene'].unique())
		gene_end = int(*gene_df['End_gene'].unique())

		# assert that the correct chromosome fasta entry is loaded
		while chrom_id != chrom:
			chrom_id, chrom_seq = next(fasta_generator)
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
		# TODO: calibrate downstram window
		emb_start, emb_end = gene_start - up_stream_window, gene_start + down_stream_window
		emb_length = emb_end - emb_start

		peaks = [(int(s), int(e))for s, e in gene_df[['Start_peak', 'End_peak']].to_numpy()]

		# create embedding parts
		embedding = [f(
			dna=chrom_seq[emb_start:emb_end],
			embedding_interval=(emb_start, emb_end),
			peak_intervals=peaks,
			# TODO: CNV data
			# TODO: CDS data
			) for f in parts]
		
		# combine embedding to one numpy array
		embedding = np.vstack(embedding)

		# TODO how to return by sample or by gene?
		yield embedding


