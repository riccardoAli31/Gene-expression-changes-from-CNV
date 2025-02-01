#!/bin/python

"""
Script for creating embeddings from DNA regions.
See https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x/figures/5 
"""

from numpy import array, ndarray, hstack, vstack, zeros, uint8
from pandas import DataFrame, Series, read_csv, concat
from typing import List, Tuple, Union, Set, Generator, Any
import os
from ..data import (
    allosomes, 
    autosomes,
    extract_cnv_overlaps,
    generate_fasta_entries,
    dna_padding
)
from ..util import relative_idx


# TODO:
#	- how to handle missing CNV overlaps
#	- read reference genome from .fasta.tar.gz using import tarfile?
#	- use data .gtf file for CDS and promoters

# TODO: implement two different modes:
# - one returning multiple genes concatenated per cell
# - another returning multiple genes as channels for each gene

# Possible bugs:
# - close to chromosomal end -> no epiAneufinder data

embedding_modes = {'gene_concat', 'single_gene_barcode', 'barcode_channel'}

def encode_dna_seq(dna: str, **kwargs) -> ndarray:
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
		"A": array([1, 0, 0, 0], dtype='u1', ndmin=2).T,
		"C": array([0, 1, 0, 0], dtype='u1', ndmin=2).T,
		"G": array([0, 0, 1, 0], dtype='u1', ndmin=2).T,
		"T": array([0, 0, 0, 1], dtype='u1', ndmin=2).T,
		# "U": array([0, 0, 0, 1], dtype='u1', ndmin=2).T,
		"R": array([1, 0, 1, 0], dtype='u1', ndmin=2).T,
		"Y": array([0, 1, 0, 1], dtype='u1', ndmin=2).T,
		"M": array([1, 1, 0, 0], dtype='u1', ndmin=2).T,
		"K": array([0, 0, 1, 1], dtype='u1', ndmin=2).T,
		"S": array([0, 1, 1, 0], dtype='u1', ndmin=2).T,
		"W": array([1, 0, 0, 1], dtype='u1', ndmin=2).T,
		"B": array([0, 1, 1, 1], dtype='u1', ndmin=2).T,
		"D": array([1, 0, 1, 1], dtype='u1', ndmin=2).T,
		"H": array([1, 1, 0, 1], dtype='u1', ndmin=2).T,
		"V": array([1, 1, 1, 0], dtype='u1', ndmin=2).T,
		"N": array([1, 1, 1, 1], dtype='u1', ndmin=2).T,
		# "X": array([1, 1, 1, 1], dtype='u1', ndmin=2).T,
		# "-": array([0, 0, 0, 0], dtype='u1', ndmin=2).T,
	}

	# encoded_dna = zeros((len(alphabet), len(dna)), dtype='u1')
	encoded_dna = [nucleotide_encoding[aa] for aa in dna]
	return hstack(encoded_dna)


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
	atac_embedding = zeros((1, emb_end - emb_start), dtype='u1')
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
					  **kwargs) -> ndarray:
	"""
	The CNV embedding part is a (2 x embedding_length) numpy.ndarray where the
	first row encodes a CNV loss and the second row CNS gains.
	The encoding for both rows works as (0: no loss/gain, 1: loss/gain)
	Thus, the gain vector will be 0 at all normal or loss positions and
	 vice versa.

	embedding_interval : (int, int) of chromosomal embedding start and stop
	cnv_interval_status : ((int, int), int) of copy number intervals, where the
		nested tuple specifies chromosomal start and end of the interval and the
		last integer encodes the copy number status, as follows:
		0: 'loss', 1: 'normal', 2: 'gain'
	"""

	emb_start, emb_end = embedding_interval
	emb_length = emb_end - emb_start

	# TODO: how to handle missing values / CNV overlaps?
	if cnv_interval_status == [((),)]:
		return zeros((2, emb_length))

	cnv_loss = zeros(emb_length)
	cnv_gain = zeros(emb_length)

	for (start, end), status in cnv_interval_status:
		start = relative_idx(start, embedding_interval)
		end = relative_idx(end, embedding_interval)
		if status == 0:
			cnv_loss[start:end] = 1
		elif status == 2:
			cnv_gain[start:end] = 1

	return vstack([cnv_loss, cnv_gain])


def embed_dna(gene_regions: DataFrame, embedding_window: Tuple[int,int],
			  fasta_path: str, pad_dna=True, verbose=False
			  ) -> Generator[Tuple[str,int,int,ndarray],Any,Any]:
	"""
	DNA sequence embedding generator.
	This wrapper generates DNA sequence embeddings from the reference genome for
	each chromosomal embedding position.

	gene_regions : pandas.DataFrame of chrom, strat, end and ENSEMBL id for all
		genes of interest
	embedding_window : (int, int) of chromosomal embedding start and end
	fasta_path : str path to reference genome .fasta file.
	pad_dna : bool wether to use padding character for embedding positions 
		downstream of the associated gene end position.
	verbose : bool wether to print file reading progress.
	"""

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
	

def embed_atac(gene_regions: DataFrame, embedding_window: Tuple[int,int],
			   atac_df: DataFrame
			   ) -> Generator[Tuple[str,int,int,ndarray],Any,Any]:
	"""
	Open Chromatin embedding generator.
	This wrapper generates open chromatin embeddings based on overlaps between 
	the embedding and the ATAC-seq peaks.

	gene_regions : pandas.DataFrame of chrom, strat, end and ENSEMBL id for all
		genes of interest
	embedding_window : (int, int) of chromosomal embedding start and end
	atac_df : pandas.DataFrame of overlaps between ATAC-seq peaks and genes
	"""

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


def embed_cnv(gene_regions: DataFrame, embedding_window: Tuple[int,int],
			  cnv_path: str, mode='gene_concat',
			  barcode_set: Union[Set[str], None]=None 
			  ) -> Generator[Tuple[str,int,int,ndarray],Any,Any]:
	"""
	CNV embedding generator.
	This wrapper generates CNV embeddings based on the specified mode.

	gene_regions : pandas.DataFrame of chrom, strat, end and ENSEMBL id for all
		genes of interest
	embedding_window : (int, int) of chromosomal embedding start and end
	cnv_path : str path to EpiAneufinder result_table.tsv
	mode : str one of {'gene_concat', 'single_gene_barcode', 'barcode_channel'}
		embedding generation modus. Default: 'gene_concat'
	barcode_set : {str} or None specifying cell barcode subset to use
	"""

	assert mode in embedding_modes
	emb_upstream, emb_downstream = embedding_window

	# load EpiAneufinder result table 
	cnv_df = read_csv(cnv_path, sep=' ')
	cnv_df = cnv_df.sort_values(by=['seq', 'start', 'end'])
	assert all(cnv_df.columns[1:4] == ['seq', 'start', 'end']),\
		"Column name Mismatch, please insert 'idx ' to the first line."
	cnv_df['seq'] = Series(map(lambda x: x.replace('chr', ''), cnv_df['seq']))

	if barcode_set is not None:
		assert len(barcode_set.difference(set(cnv_df.columns[4:]))) == 0,\
			"Barcodes don't match CNV data!"
		cnv_df = cnv_df[
			cnv_df.columns[
				cnv_df.columns.isin(barcode_set.union({'seq', 'start', 'end'}))
			]
		]

	match mode:
		case 'gene_concat' | 'single_gene_barcode':
			for barcode in cnv_df.columns[3:]:
				for _, (chrom, gen_start, _, gen_id) in gene_regions.iterrows():
					emb_start = gen_start - emb_upstream
					emb_end = gen_start + emb_downstream
					yield (
						barcode,
						gen_id,
						encode_cnv_status(
							(emb_start, emb_end),
							extract_cnv_overlaps(
								cnv_df=cnv_df,
								barcode=barcode,
								region=(chrom, emb_start, emb_end)
							)
						)
					)
		
		case 'barcode_channel':
			for _, (chrom, gen_start, _, gen_id) in gene_regions.iterrows():
				for barcode in cnv_df.columns[cnv_df.columns.isin(barcode_set)]:
					emb_start = gen_start - emb_upstream
					emb_end = gen_start + emb_downstream
					yield (
						barcode,
						gen_id,
						encode_cnv_status(
							(emb_start, emb_end),
							extract_cnv_overlaps(
								cnv_df=cnv_df,
								barcode=barcode,
								region=(chrom, emb_start, emb_end)
							)
						)
					)
	

def embed(fasta_path, atac_path, cnv_path, gene_set: Union[Set[str], None],
		  barcode_set: Union[Set[str], None], mode='gene_concat', pad_dna=True,
		  n_upstream=2000, n_downstream=8000, gtf_path=None
		  ) -> Generator[ndarray,Any,Any]:
	"""
	Main wrapper function. Generates embeddings from following files:
	* fasta reference genome sequence
	* overlap data of ATAC-seq peaks and genes
	* CNV results from from EpiAneufinder (add column name for index in 1st row)
	* gtf genome annotation (optional, only if CDS, promoter, ... data wanted) 

	fasta_path : str path to reference genome .fasta file
	atac_path : str path to overlap data from ATAC-seq peaks and genes (.tsv)
	cnv_path : str path to results from from EpiAneufinder (.tsv file)
	gene_set : Set[str] or None specifiying subset of ENSEMBL gene ids to use
	barcode_set : Set[str] or None specifying subset of cell barcodes to use
	mode : str one of ['gene_concat', 'barcode_channel', 'single_gene_barcode']
		to specify how to combine and return embeddings. Default: 'gene_concat'
	pad_dna : bool if pad embedding positions after gene end. Default: True
	n_upstream : int of embedding length upstream of gene start. Default: 2000
	n_downstream : int of embedding len. downstram of gene start. Default: 8000
	gtf_path : None or str path to .gtf annotation file 
	"""

	assert all(map(os.path.isfile, [fasta_path, atac_path, cnv_path]))

	# load open chromatin peaks
	atac_df = read_csv(atac_path, sep='\t')
	# sort autosomes on integer index
	atac_df_auto = atac_df[atac_df['Chromosome'].isin(autosomes)].copy()
	atac_df_auto['Chromosome'] = atac_df_auto['Chromosome'].astype(uint8)
	atac_df_auto = atac_df_auto.sort_values(by=['Chromosome', 'Start_gene', 'End_gene'])
	# sort allosomes separately
	atac_df_allo = atac_df[atac_df['Chromosome'].isin(allosomes)].copy()
	atac_df_allo = atac_df_allo.sort_values(by=['Chromosome', 'Start_gene', 'End_gene'])
	atac_df_auto['Chromosome'] = atac_df_auto['Chromosome'].astype(str)
	# concat sorted dataframes
	atac_df = concat([atac_df_auto, atac_df_allo])
	uniq_gene_ids = list(atac_df['gene_id'].unique())

	# apply subsetting by gene_id
	if gene_set is not None:
		uniq_gene_ids = set(uniq_gene_ids).intersection(gene_set)

	gene_df = atac_df[atac_df['gene_id'].isin(uniq_gene_ids)][['Chromosome', 'Start_gene', 'End_gene', 'gene_id']].drop_duplicates()

	# create embedding part generators
	# TODO: divide into barcode, dependent and barcode independent and create using list of funcitons
	dna_embedder = embed_dna(
		gene_regions=gene_df,
		embedding_window=(n_upstream, n_downstream),
		fasta_path=fasta_path,
		pad_dna=pad_dna
	)
	atac_embedder = embed_atac(
		gene_regions=gene_df,
		embedding_window=(n_upstream, n_downstream),
		atac_df=atac_df
	)
	cnv_embedder = embed_cnv(
		gene_regions=gene_df,
		embedding_window=(n_upstream, n_downstream),
		cnv_path=cnv_path,
		barcode_set=barcode_set,
		mode=mode
	)

	genomic_embeddings = []
	barcode_embeddings = []
	# barcode_embeddings.append(cnv_embedding)
	for _, (chrom, gene_start, gene_end, gene_id) in gene_df.iterrows():
		
		_, _, _, dna_embedding = next(dna_embedder)
		_, _, _, atac_embedding = next(atac_embedder)

		genomic_embedding = vstack([
			dna_embedding,
			atac_embedding
		])

		genomic_embeddings.append(genomic_embedding)

		barcode, cnv_gene_id, cnv_embedding = next(cnv_embedder)

		match mode:
			case 'barcode_channel':
				# TODO: debug
				while cnv_gene_id == gene_id:
					print(gene_id, barcode)
					barcode_embeddings.append(cnv_embedding)
					barcode, cnv_gene_id, cnv_embedding = next(cnv_embedder)
				# TODO: repreat genomic embedding len(barcode_embeddings) times
				# numpy.tile(genomic_embedding, (*genomic_embedding.shape, len(barcode_embeddings)))
				yield vstack([genomic_embedding, vstack(barcode_embeddings)])
				barcode_embeddings = [cnv_embedding]
				continue
			case 'single_gene_barcode':
					yield vstack([genomic_embedding, cnv_embedding])
			case 'gene_concat':
					assert gene_id == cnv_gene_id
					barcode_embeddings.append(cnv_embedding)

	if mode == 'gene_concat':
		# barcode_embeddings.append(cnv_embedding)
		print(barcode, '1:', len(barcode_embeddings))
		yield vstack([
			hstack(genomic_embeddings),
			hstack(barcode_embeddings)
		])
		barcode_embeddings = []

		# generate all barcode embeddings for remaining barcodes
		barcode, cnv_gene_id, cnv_embedding = next(cnv_embedder)
		for next_barcode, next_cnv_gene_id, next_cnv_embedding in cnv_embedder:
			if barcode != next_barcode:
				print("switsching barcode")
				barcode_embeddings.append(cnv_embedding)
				yield vstack([
					hstack(genomic_embeddings),
					hstack(barcode_embeddings)
				])
				barcode_embeddings = []
			else:
				barcode_embeddings.append(cnv_embedding)
			
			barcode, cnv_gene_id, cnv_embedding = next_barcode, next_cnv_gene_id, next_cnv_embedding	
		
		# handle fence post
		print("reaching fence post, len(barcode)", len(barcode_embeddings))
		barcode_embeddings.append(cnv_embedding)
		yield vstack([
			hstack(genomic_embeddings),
			hstack(barcode_embeddings)
		])

