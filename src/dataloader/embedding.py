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
from ..data import allosomes, autosomes, extract_cnv_overlaps, generate_fasta_entries, dna_padding
from ..util import relative_idx


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


# @PendingDeprecationWarning
# def encode_cds_structure(cds_intervals: List[Tuple[int, int]], seq_len: int, **kwargs) -> ndarray:
# 	# TODO
# 	cds_structure = np.zeros(seq_len, dtype='u1')
# 	for start, stop in cds_intervals:
# 		cds_structure[start:stop] = 1

# 	return cds_structure


# @PendingDeprecationWarning
# def encode_reading_frame(CDS_start_stop: List[Tuple[int, int]],seq_len: int, **kwargs):
# 	# TODO compute reading frame from CDS structure
# 	codon_size = 3
# 	reading_frame = np.zeros(seq_len, dtype='u1')
# 	overhead = 0
# 	for start, stop in CDS_start_stop:
# 		reading_frame[list(start + (codon_size - overhead), stop, codon_size)] = 1
# 		overhead = (stop - start + overhead) % 3

# 	return reading_frame


def embed_dna(gene_regions: pd.DataFrame, fasta_path: str,
		embedding_window: Tuple[int,int], pad_dna=True, verbose=False):
	# generate DNA embeddings
	assert os.path.isfile(fasta_path), "FASTA file not found: {}".format(fasta_path)
	fasta_generator = generate_fasta_entries(fasta_path)
	emb_upstream, emb_downstream = embedding_window

	chrom_id, chrom_seq = next(fasta_generator)
	print("chrom_id:", chrom_id) if verbose else None
	for _, (chrom, gene_start, gene_end, _) in gene_regions.iterrows():
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
	

def embed_atac(gene_regions: pd.DataFrame, atac_df: pd.DataFrame, embedding_window: Tuple[int,int]):
	emb_upstream, emb_downstream = embedding_window
	for _, (chrom, gene_start, gene_end, _) in gene_regions.iterrows():
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


def embed_cnv(gene_regions: pd.DataFrame, cnv_path: str,
		embedding_window: Tuple[int,int], mode='gene_concat',
		barcode_set: Union[Set[str], None]=None):

	emb_upstream, emb_downstream = embedding_window

	# load copy number data
	cnv_df = pd.read_csv(cnv_path, sep=' ')
	cnv_df = cnv_df.sort_values(by=['seq', 'start', 'end'])
	cnv_df['seq'] = pd.Series(map(lambda x: x.replace('chr', ''), cnv_df['seq']))

	if barcode_set is not None:
		assert len(barcode_set.difference(set(cnv_df.columns[4:]))) == 0, "Barcodes don't match CNV data!"
		cnv_df = cnv_df[cnv_df.columns[cnv_df.columns.isin(barcode_set.union({'seq', 'start', 'end'}))]]

	match mode:
		case 'gene_concat' | 'single_gene_barcode':
			for barcode in cnv_df.columns[4:]:
				for _, (chrom, gene_start, _, gene_id) in gene_regions.iterrows():
					emb_start, emb_end = gene_start - emb_upstream, gene_start + emb_downstream
					yield (
						barcode,
						gene_id,
						encode_cnv_status(
							(emb_start, emb_end),
							extract_cnv_overlaps(cnv_df, barcode, (chrom, emb_start, emb_end))
						)
					)
		
		case 'barcode_channel':
			for _, (chrom, gene_start, _, gene_id) in gene_regions.iterrows():
				for barcode in cnv_df.columns[cnv_df.columns.isin(barcode_set)]:
					emb_start, emb_end = gene_start - emb_upstream, gene_start + emb_downstream
					yield (
						barcode,
						gene_id,
						encode_cnv_status(
							(emb_start, emb_end),
							extract_cnv_overlaps(cnv_df, barcode, (chrom, emb_start, emb_end))
						)
					)
	

def embed(fasta_path, atac_path, cnv_path, mode='gene_concat',
		gtf_path=None, gene_set: Union[Set[str], None]=None,
		barcode_set: Union[Set[str], None]=None,
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

	gene_df = atac_df[atac_df['gene_id'].isin(uniq_gene_ids)][['Chromosome', 'Start_gene', 'End_gene', 'gene_id']].drop_duplicates()

	# create embedding part generators
	dna_embedder = embed_dna(gene_df, fasta_path, (n_upstream, n_downstream), pad_dna=pad_dna)
	atac_embedder = embed_atac(gene_df, atac_df, (n_upstream, n_downstream))
	cnv_embedder = embed_cnv(
		gene_df,
		cnv_path,
		(n_upstream, n_downstream),
		barcode_set=barcode_set,
		mode=mode
	)

	genomic_embeddings = []
	barcode_embeddings = []
	barcode, cnv_gene_id, cnv_embedding = next(cnv_embedder)
	# barcode_embeddings.append(cnv_embedding)
	for _, (chrom, gene_start, gene_end, gene_id) in gene_df.iterrows():
		
		_, _, _, dna_embedding = next(dna_embedder)
		_, _, _, atac_embedding = next(atac_embedder)

		genomic_embedding = np.vstack([
			dna_embedding,
			atac_embedding
		])

		genomic_embeddings.append(genomic_embedding)

		next_barcode, next_cnv_gene_id, next_cnv_embedding = next(cnv_embedder)

		if mode == 'barcode_channel':
			# TODO: debug
			while cnv_gene_id == gene_id:
				print(gene_id, barcode)
				barcode_embeddings.append(cnv_embedding)
				barcode, cnv_gene_id, cnv_embedding = next_barcode, next_cnv_gene_id, next_cnv_embedding
				next_barcode, next_cnv_gene_id, next_cnv_embedding = next(cnv_embedder)
			# TODO: repreat genomic embedding len(barcode_embeddings) times
			# numpy.tile(genomic_embedding, (*genomic_embedding.shape, len(barcode_embeddings)))
			yield np.vstack([genomic_embedding, np.vstack(barcode_embeddings)])
			barcode_embeddings = [cnv_embedding]
			barcode, cnv_gene_id, cnv_embedding = next_barcode, next_cnv_gene_id, next_cnv_embedding
			continue
		if mode == 'single_gene_barcode':
				yield np.vstack([genomic_embedding, cnv_embedding])
		elif barcode is not None and next_barcode != barcode:
			# TODO: can this even be reached?
			if mode == 'gene_concat':
				barcode_embeddings.append(cnv_embedding)
				yield np.vstack([
					np.hstack(genomic_embeddings),
					np.hstack(barcode_embeddings)
				])
				barcode_embeddings = []
		else:
			if mode == 'gene_concat':
				barcode_embeddings.append(cnv_embedding)
		
		barcode, cnv_gene_id, cnv_embedding = next_barcode, next_cnv_gene_id, next_cnv_embedding
	
	if mode == 'barcode_channel':
		return
	elif mode == 'gene_concat':
		# TODO: generate all barcode embeddings for noxt barcode 
		for next_barcode, next_cnv_gene_id, next_cnv_embedding in cnv_embedder:
			if barcode != next_barcode:
				barcode_embeddings.append(cnv_embedding)
				yield np.vstack([
					np.hstack(genomic_embeddings),
					np.hstack(barcode_embeddings)
				])
				barcode_embeddings = []
			else:
				barcode_embeddings.append(cnv_embedding)
			
			barcode, cnv_gene_id, cnv_embedding = next_barcode, next_cnv_gene_id, next_cnv_embedding	
		
		# handle fence post
		barcode_embeddings.append(cnv_embedding)
		yield np.vstack([
			np.hstack(genomic_embeddings),
			np.hstack(barcode_embeddings)
		])

