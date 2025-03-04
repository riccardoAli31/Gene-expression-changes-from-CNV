#!/bin/python

"""
Script for creating embeddings from DNA regions.
See https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x/figures/5 
"""

from pathlib import Path
from numpy import array, ndarray, hstack, vstack, tile, zeros, ones, uint8
from numpy import concat as np_concat
from pandas import DataFrame, Series, read_csv
from pandas import concat as pd_concat
from typing import List, Tuple, Dict, Union, Set, Generator, Any
import pyranges as pr
from tqdm import tqdm
from warnings import warn
import os
from . import (
    allosomes, 
    autosomes,
    extract_cnv_overlaps,
    generate_fasta_entries,
    dna_padding
)
from . import standard_chromosomes
from ..util import relative_idx


# TODO:
#	- how to handle missing CNV overlaps
#	- read reference genome from .fasta.tar.gz using import tarfile?
#	- use data .gtf file for CDS and promoters
#	* divide into barcode, dependent and barcode independent embedders
#	* create using list of funcitons
#	* use pyranges for overlap computation https://pyranges.readthedocs.io/en/latest/autoapi/pyranges/index.html
#	* cite pyranges https://academic.oup.com/bioinformatics/article/36/3/918/5543103
#	* use ICLR conference paper template for report

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
	
	dtype = kwargs.get('dtype', 'u1')
	dna = dna.upper()

	nucleotide_encoding = {
		"A": array([1, 0, 0, 0], dtype=dtype, ndmin=2).T,
		"C": array([0, 1, 0, 0], dtype=dtype, ndmin=2).T,
		"G": array([0, 0, 1, 0], dtype=dtype, ndmin=2).T,
		"T": array([0, 0, 0, 1], dtype=dtype, ndmin=2).T,
		# "U": array([0, 0, 0, 1], dtype=dtype, ndmin=2).T,
		"R": array([1, 0, 1, 0], dtype=dtype, ndmin=2).T,
		"Y": array([0, 1, 0, 1], dtype=dtype, ndmin=2).T,
		"M": array([1, 1, 0, 0], dtype=dtype, ndmin=2).T,
		"K": array([0, 0, 1, 1], dtype=dtype, ndmin=2).T,
		"S": array([0, 1, 1, 0], dtype=dtype, ndmin=2).T,
		"W": array([1, 0, 0, 1], dtype=dtype, ndmin=2).T,
		"B": array([0, 1, 1, 1], dtype=dtype, ndmin=2).T,
		"D": array([1, 0, 1, 1], dtype=dtype, ndmin=2).T,
		"H": array([1, 1, 0, 1], dtype=dtype, ndmin=2).T,
		"V": array([1, 1, 1, 0], dtype=dtype, ndmin=2).T,
		"N": array([1, 1, 1, 1], dtype=dtype, ndmin=2).T,
		# "X": array([1, 1, 1, 1], dtype=dtype, ndmin=2).T,
		# "-": array([0, 0, 0, 0], dtype=dtype, ndmin=2).T,
	}

	# encoded_dna = zeros((len(alphabet), len(dna)), dtype=dtype)
	encoded_dna = [nucleotide_encoding[aa] for aa in dna]
	return hstack(encoded_dna)


def encode_open_chromatin(embedding_interval: Tuple[int, int], 
						  peak_intervals: List[Tuple[int, int]],
						  verbose=False, **kwargs) -> ndarray:
	"""
	Creates the open chromatin embedding part, that observes ones all
	genmoic positions that are in open chromatin and 0 for closed.

	embedding_interval: Tuple[int,int] of genomic start and end for the 
		region to create an embedding for
	peak_intervals : List[Tuple[int,int]] of start and end position for a
		list of peaks from ATAC-seq  
	"""

	dtype = kwargs.get('dtype', 'u1')
	emb_start, emb_end = embedding_interval
	assert emb_start < emb_end
	atac_embedding = zeros((1, emb_end - emb_start), dtype=dtype)
	for (peak_start, peak_end) in peak_intervals:
		if emb_end <= peak_start:
			continue
		if emb_end < peak_end:
			peak_end = emb_end

		# sanity check there is an overlap between embedding and peak
		if not (peak_start < emb_end and emb_start < peak_end):
			# TODO: this means missing data, encode this diffenently?
			if verbose:
				print(
					'[encode_open_chromatin] No peak overlap!\n' + 
					': peak: {}-{}, embedding:{}-{}'.format(
						peak_start, peak_end, emb_start, emb_end
					)
				)
			return atac_embedding

		peak_start = emb_start if peak_start < emb_start else peak_start
		peak_end = peak_end if peak_end < emb_end else emb_end

		# calculate relative start stop on embedding
		atac_embedding[peak_start - emb_start:peak_end - emb_start] = 1
		
	return atac_embedding


def encode_cnv_status(embedding_region: pr.PyRanges, cnv_regions: pr.PyRanges,
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

	assert len(embedding_region) == 1, 'Expecting only one genomic range!'
	assert len(cnv_regions.columns) == 4, 'Expecting 4 column cnv status data!'

	dtype = kwargs.get('dtype', 'u1')
	emb_start, emb_end = embedding_region.Start, embedding_region.End
	emb_length = emb_end - emb_start

	# TODO: how to handle missing values / CNV overlaps?
	if cnv_regions.empty:
		raise RuntimeError('No CNV data for {} and {} at {}:{}-{}'.format(
			embedding_region.gene_id.iloc[0], embedding_region.barcode.iloc[0], 
			embedding_region.Chromosome.iloc[0], emb_start, emb_end
		))
		# return zeros((2, emb_length))

	cnv_loss = zeros(emb_length, dtype=dtype)
	cnv_gain = zeros(emb_length, dtype=dtype)

	for _ ,(chromosome, start, end, status) in cnv_regions.df.iterrows():
		# start and end are already calculated by PyRanges.intersect
		# start = relative_idx(start, (emb_start, emb_end))
		# end = relative_idx(end, (emb_start, emb_end))
		if status == 0:
			cnv_loss[start:end] = 1
		elif status == 2:
			cnv_gain[start:end] = 1

	return vstack([cnv_loss, cnv_gain])


@DeprecationWarning
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

	TODO: use pyranges.get_sequence()
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


def get_dna_embedding(regions: pr.PyRanges, fasta_path:str, 
					  pad_dna=True, get_seq=False, **kwargs):
	"""
	fasta_path default 'data/reference/GRCh38.d1.vd1.fa'
	"""
	assert regions.df.shape[0] == 1
	# chrom, gene_start, gene_end, _ = regions
	# emb_upstream, emb_downstream = embedding_window
	# emb_start, emb_end = gene_start - emb_upstream, gene_start + emb_downstream
	dna_seq = None
	if get_seq:
		dna_seq = pr.get_sequence(pr.PyRanges(regions), fasta_path)[0]
	else:
		dna_seq = regions.Sequence.values[0]
	if pad_dna:
		dna_seq = dna_padding(
			dna_seq,
			relative_idx(
				regions.df.iloc[0]['Gene_End'],
				(regions.df.iloc[0]['Start'], regions.df.iloc[0]['End'])
			)
		)
	return encode_dna_seq(dna_seq, **kwargs)
	
@DeprecationWarning
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


def get_atac_embedding(regions: pr.PyRanges, embedding_window: Tuple[int,int],
					   **kwargs):
	if regions.empty:
		return zeros((1, sum(embedding_window)), **kwargs)
	elif regions.df.shape[0] == 1:
		return ones((1, sum(embedding_window)), **kwargs)
	else:
		raise NotImplementedError(
			'Embedding at ATAC peak boundary for {} and {}'.format(
				regions.gene_id.iloc[0]
			)
		)
		# TODO: check what if embedding spans boundary

@DeprecationWarning
def embed_cnv(gene_regions: DataFrame, cnv_df: DataFrame,
			  embedding_window: Tuple[int,int], mode='gene_concat',
			  verbose=False,
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
	# TODO: adapt documentation

	assert mode in embedding_modes
	emb_upstream, emb_downstream = embedding_window

	match mode:
		case 'gene_concat':
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
		
		case 'barcode_channel' | 'single_gene_barcode':
			assert gene_regions.shape[0] == 1,\
				"Expecting only one row/gene in this mode got {}".format(
					gene_regions.shape[0]
				)
			chrom, gen_start, _, gen_id = gene_regions.iloc[0]
			if verbose:
				print('[embed_cnv]: cnv_df.columns[3:]', cnv_df.columns[3:])
			for barcode in cnv_df.columns[3:]:
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


def get_cnv_embedding(region: pr.PyRanges, cnv_pr: pr.PyRanges, barcode: str,
					  mode='single_gene_barcode', verbose=False, **kwargs):
	assert mode == 'single_gene_barcode'
	assert region.df.shape[0] == 1,\
		'Expecting only one row/gene in this mode got {}'.format(
			region.df.shape[0]
		)
	assert barcode in cnv_pr.columns[3:],\
		'Barcode {} not found in CNV data frame for gene {}'.format(
			barcode, region.gene_id.iloc[0]
		)

	if verbose:
		print('[embed_cnv]: cnv_df.columns[3:]', cnv_pr.columns[3:])

	# chrom, emb_start, emb_end, gen_id = region.df.iloc[0]
	
	return encode_cnv_status(
		embedding_region=region,
		cnv_regions=cnv_pr[[barcode]].intersect(region),
		**kwargs
	)


def embed(fasta_path, gtf_path, atac_path, cnv_path, mode='single_gene_barcode',
		  barcode_to_genes: Union[Dict[str, List[str]], None]=None,
		  barcode_set: Union[Set[str], None]=None,
		  gene_set: Union[Set[str], None]=None, verbose=False,
		  pad_dna=True, n_upstream=2000, n_downstream=8000,
		  ) -> Generator[Tuple[str,str,ndarray],Any,Any]:
	"""
	Main wrapper function. Generates embeddings from following files:
	* fasta reference genome sequence
	* overlap data of ATAC-seq peaks and genes
	* CNV results from from EpiAneufinder (add column name for index in 1st row)
	* gtf genome annotation (optional, only if CDS, promoter, ... data wanted)

	Embeddings will be returned in genomic gene order (1st gene on chr1,
	2nd gene on chr1, ... last gene on chrY) and alphabetical barcode order.

	fasta_path : str path to reference genome .fasta file
	atac_path : str path to overlap data from ATAC-seq peaks and genes (.tsv)
	cnv_path : str path to results from from EpiAneufinder (.tsv file)
	barcode_to_genes : Dict[str, List[str]] of ENSEMBL gene id lists (values)
		for different barcodes (keys).
	gene_set : Set[str] or None specifiying subset of ENSEMBL gene ids to use
	barcode_set : Set[str] or None specifying subset of cell barcodes to use
	mode : str one of ['gene_concat', 'barcode_channel', 'single_gene_barcode']
		to specify how to combine and return embeddings. Output shapes:
		* 'gene_concat': (n_coding_rows, embedding_len * n_genes) per barcode
		* 'barcode_channel': (n_barcodes, n_coding_rows, embedding_len) per gene
		* 'single_gene_barcode': (n_coding_rows, embedding_len) per gene - barc.
		Default: 'gene_concat'
	pad_dna : bool if pad embedding positions after gene end. Default: True
	n_upstream : int of embedding length upstream of gene start. Default: 2000
	n_downstream : int of embedding len. downstram of gene start. Default: 8000
	gtf_path : None or str path to .gtf annotation file

	returns : Tuple[str,str,ndarray] of cell/barcode id, gene id and embedding
		as numpy.ndarray. Note depending on mode the value of gene_id or 
		barcode_id might be 'all barcodes' (in case of 'barcode_channel') or 
		'all genes' ('gene_concat'). This is due to the fact, that in these
		modes all barcodes or genes are represented in the returned embedding.
	"""

	assert os.path.isfile(fasta_path), 'FASTA not found: {}'.format(fasta_path)
	assert os.path.isfile(gtf_path), 'GTF not found: {}'.format(gtf_path)
	assert os.path.isfile(atac_path), 'Overlaps not found: {}'.format(atac_path)
	assert os.path.isfile(cnv_path), 'CNV file not found: {}'.format(cnv_path)
	assert mode == 'single_gene_barcode', \
		'Only supporting "single_gene_barcode" mode at the moment!'
	
	# TODO: use pyranges to parse gene entries
	# * filter genes not present in gtf -> raise exception
	# * quick fix: use gene positions and overlap file from gaja
	# import pyranges as pr
	# gtf = pr.read_gtf('data/reference/Homo_sapiens.GRCh38.113.gtf.gz')
	# import sys
	# sys.getsizeof(gtf)
	# gtf[(gtf.Feature == 'gene') & (gtf.Chromosome.isin(['1', '21']))].df.loc[0]
	# pr.get_sequence(gtf[(gtf.Feature == 'gene') & (gtf.gene_id == 'ENSG00000142611')], 'data/reference/GRCh38.d1.vd1.fa')
	# use from pympler import asizeof for memory checks

	# TODO:
	# 1. read gft/genes file and filter for genes in gene_set
	# 2. compute embedding start and end from gene start / end
	# 3. switch start and end columns with emb start and end
	# 4. convert to PyRanges
	# 5. get sequences from fasta
	# 6. create data table with all wanted gene-barcode combinations
	# 7. iterate over gene_id, barcode pairs
	#   7.1. get ATAC per barcode
	#	7.2. get CNVs per barcode

	# === GTF ANNOTATION ===
	# TODO: use pyranges to read annotation from gtf: 
	# for now use 'data/gene_positions_and_overlaps/gene_positions.csv'
	gene_df = read_csv(gtf_path)
	gene_df = gene_df.rename(columns={
		'seqnames': 'Chromosome', 'start': 'Gene_Start', 'end': 'Gene_End'
	})
	gene_df = gene_df[gene_df['Chromosome'].isin(standard_chromosomes)]
	uniq_gene_ids = set(gene_df['gene_id'])

	# apply subsetting by gene_id
	if gene_set is not None:
		uniq_gene_overlap = uniq_gene_ids.intersection(gene_set)
		if len(gene_set) - len(uniq_gene_overlap) > 0:
			print('[embed]: No GTF annot. for {} genes from gene_set'.format(
				len(gene_set) - len(uniq_gene_overlap)
			))
		if verbose:
			print(','.join(
				[gid for gid in gene_set if gid not in uniq_gene_overlap]
			))
		uniq_gene_ids = uniq_gene_overlap
	if barcode_to_genes is not None:
		iter_gene_set = {
			gene for vals in barcode_to_genes.values() for gene in vals
		}
		uniq_gene_overlap = uniq_gene_ids.intersection(iter_gene_set)
		if len(iter_gene_set) - len(uniq_gene_overlap) > 0:
			print('[embed]: No GTF annot. for {} genes from barcode_to_genes'\
				.format(len(iter_gene_set) - len(uniq_gene_overlap))
			)
		if verbose:
			print(','.join(
				[gid for gid in iter_gene_set if gid not in uniq_gene_overlap]
			))
		uniq_gene_ids = uniq_gene_overlap

	gene_df = gene_df[gene_df['gene_id'].isin(uniq_gene_ids)]

	# calculate embedding starts
	gene_df['Start'] = gene_df['Gene_Start'] - n_upstream
	gene_df['End'] = gene_df['Gene_Start'] + n_downstream

	# convert to pyranges
	gene_df = pr.PyRanges(gene_df)
	gene_df.Sequence = pr.get_sequence(gene_df, fasta_path)
	print('[embed]:', gene_df)

	# ==== OPEN CHROMATIN ====
	# load open chromatin peaks
	atac_df = read_csv(atac_path, sep='\t')
	atac_df = atac_df.rename(columns={
		'Start_peak': 'Start', 'End_peak': 'End'
	})
	atac_df = pr.PyRanges(atac_df)
	# # sort autosomes on integer index
	# atac_df_auto = atac_df[atac_df['Chromosome'].isin(autosomes)].copy()
	# atac_df_auto['Chromosome'] = atac_df_auto['Chromosome'].astype(uint8)
	# atac_df_auto = atac_df_auto.sort_values(
	# 	by=['Chromosome', 'Start_gene', 'End_gene']
	# )
	# # sort allosomes separately
	# atac_df_allo = atac_df[atac_df['Chromosome'].isin(allosomes)].copy()
	# atac_df_allo = atac_df_allo.sort_values(
	# 	by=['Chromosome', 'Start_gene', 'End_gene']
	# )
	# atac_df_auto['Chromosome'] = atac_df_auto['Chromosome'].astype(str)
	# # concat sorted dataframes
	# atac_df = pd_concat([atac_df_auto, atac_df_allo])
	
	# ==== CNV ====
	# load EpiAneufinder result table 
	cnv_df = read_csv(cnv_path, sep=' ')
	assert all(cnv_df.columns[1:4] == ['seq', 'start', 'end']),\
		'Column name Mismatch, please prepend \'idx \' to the first line.'
	
	# define unique barcodes to use for embedding computation
	uniq_barcodes = set(cnv_df.columns[4:])
	if barcode_set is not None:
		barcode_diff = barcode_set.difference(uniq_barcodes)
		if len(barcode_diff) > 0:
			warn('[embed]: No CNV data for {}'.format(','.join(barcode_diff)))
		uniq_barcodes = uniq_barcodes.intersection(barcode_set)

	# remove not needed barcodes
	cnv_df = cnv_df[
		cnv_df.columns[
			cnv_df.columns.isin(uniq_barcodes.union({'seq', 'start', 'end'}))
		]
	]

	# cnv_df = cnv_df.sort_values(by=['seq', 'start', 'end'])
	cnv_df = cnv_df.astype({c: 'u1' for c in cnv_df.columns[4:]})
	cnv_df['seq'] = Series(map(lambda x: x.replace('chr', ''), cnv_df['seq']))
	cnv_df = cnv_df.rename(columns={
		'seq': 'Chromosome', 'start': 'Start', 'end': 'End'
	})
	# print('[embed]:', cnv_df)
	# cnv_df = cnv_df.drop('idx', axis=1)
	cnv_df = pr.PyRanges(cnv_df)
	#cnv_df = cnv_df.overlap(gene_df) # only retain overlaps with relevant genes
	cnv_df = gene_df.join(cnv_df) # do inner join
	# TODO: test overlaps at 100k points
	# gc_join = gene_pr.join(cnv_pr)
	# gc_join.df.apply(lambda r: int(str(r['Start'])[0]) < int(str(r['End'])[0]), axis = 1)
	print('[embed]:', cnv_df)

	# ==== Iteration Mapping ====
	# create list of tuples based on barcode_to_genes dict
	gene_barcode_pairs = list()
	n_embeddings = 0
	if barcode_to_genes is not None:
		n_embeddings = len([
			g for gl in barcode_to_genes.values() for g in gl 
			if g in uniq_gene_ids
		])
		print('[embed]: Iterating over custom barcode to genes mapping')
		iter_barcode_set = set(barcode_to_genes.keys())
		uniq_barcodes = uniq_barcodes.intersection(iter_barcode_set)

		# sort uniq barcodes for alphabetical iteration order
		uniq_barcodes = sorted(list(uniq_barcodes))

		# create a mapping from gene to barcode to serve for iteration later
		gene_to_barcodes = {gene: list() for gene in uniq_gene_ids}
		# TODO: use boolean map to encode barcodes per gene for saving memory
		for cnv_barcode in uniq_barcodes:
			for gene in barcode_to_genes[cnv_barcode]:
				if gene in uniq_gene_ids:
					gene_to_barcodes[gene].append(cnv_barcode)

		gene_barcode_pairs = [
			(g, b) for g in gene_df.df['gene_id'] for b in gene_to_barcodes[g]
		]
		
	else:
		print('[embed]: Iterating over all possible barcode-gene combinations')
		n_embeddings = len(uniq_barcodes) * len(uniq_gene_ids)

		# sort uniq barcodes for alphabetical iteration order
		uniq_barcodes = sorted(list(uniq_barcodes))

		# gene_to_barcodes = {
		# 	gene: uniq_barcodes for gene in uniq_gene_ids
		# }
		gene_barcode_pairs = [
			(g, b) for g in gene_df.df['gene_id'] for b in uniq_barcodes
		]

	if len(uniq_barcodes) == 0:
		raise RuntimeError(
			'No barcode data available!\n' +
			'Please make sure the barcodes you want to use are present in the' +
			' epiAneufinder results in order to have CNV data.'
		)
	if len(uniq_gene_ids) == 0:
		raise RuntimeError(
			'No ENSEMBL gene ids in overlap!\n' +
			'Please make sure there is both ATAC and CNV data for the genes ' +
			'you want to process.'
		)

	# for modes which return emdeddings of muliple gene or barcode size, change
	#  expected # embeddings
	# match mode:
	# 	case 'barcode_channel':
	# 		n_embeddings = len(uniq_gene_ids)
	# 	case 'gene_concat':
	# 		n_embeddings = len(uniq_barcodes)
	assert len(gene_barcode_pairs) == n_embeddings, \
		'Iteration mapping failed! Incorrect number of embeddings'

	print('[embed]: Computing {} Embeddings with mode: "{}"'.format(
		n_embeddings, mode
	))
	print('[embed]: Using {} barcodes'.format(len(uniq_barcodes)))
	if verbose:
		print('[embed]:', ','.join(uniq_barcodes))
	print('[embed]: Using {} genes'.format(len(uniq_gene_ids)))
	if verbose:
		print('[embed]:', ','.join(uniq_gene_ids))

	# # create embedding part generators
	# # TODO: 
	# # * divide into barcode, dependent and barcode independent embedders
	# # * create using list of funcitons
	# dna_embedder = embed_dna(
	# 	gene_regions=gene_df,
	# 	embedding_window=(n_upstream, n_downstream),
	# 	fasta_path=fasta_path,
	# 	pad_dna=pad_dna
	# )
	# # atac embedder only for 'single_gene_barcode' mode 
	# atac_embedder = embed_atac(
	# 	gene_regions=gene_df,
	# 	embedding_window=(n_upstream, n_downstream),
	# 	atac_df=atac_df
	# )
	# # cnv embedder only for 'single_gene_barcode' mode
	# cnv_columns = ['seq', 'start', 'end']
	# cnv_columns.extend(uniq_barcodes)
	# cnv_embedder = embed_cnv(
	# 	gene_regions=gene_df,
	# 	embedding_window=(n_upstream, n_downstream),
	# 	cnv_df=cnv_df[cnv_columns],
	# 	mode=mode
	# )

	# genomic_embeddings = []
	# barcode_embeddings = []
	embegging_iterator = tqdm(
		iter(gene_barcode_pairs), # gene_df.iterrows(),
		desc='[embed]: Computing embeddings',
		total=n_embeddings, # len(uniq_gene_ids),
		ncols=120
	)
	prev_gene_id = None
	for (gene_id, barcode) in embegging_iterator:
		# chrom, gene_start, gene_end, gene_id = row
		
		# _, _, _, dna_embedding = next(dna_embedder)
		if gene_id != prev_gene_id:
			dna_embedding = get_dna_embedding(
				gene_df[gene_df.gene_id == gene_id],
				fasta_path=fasta_path
			)
			prev_gene_id = gene_id

		# get open chromatin emebdding
		atac_embedding = get_atac_embedding(
			atac_df[
				(atac_df.gene_id == gene_id) & (atac_df.barcode == barcode)
			].overlap(gene_df),
			embedding_window=(n_upstream, n_downstream)
		)

		# get CNV embedding
		cnv_embedding = get_cnv_embedding(
			region=gene_df[gene_df.gene_id == gene_id],
			cnv_pr=cnv_df[cnv_df.gene_id == gene_id][[barcode]],
			barcode=barcode
		)

		yield (
			barcode,
			gene_id,
			vstack([dna_embedding, atac_embedding, cnv_embedding])
		)

	# 	genomic_embeddings.append(genomic_embedding)

	# 	match mode:
	# 		case 'barcode_channel':
	# 			# get barcodes selected for this gene
	# 			gene_barcode_list = gene_to_barcodes[gene_id]
	# 			cnv_columns = ['seq', 'start', 'end']
	# 			cnv_columns.extend(gene_barcode_list)
	# 			if verbose:
	# 				print('[embed]: barcode_list', gene_barcode_list)

	# 			cnv_embedder = embed_cnv(
	# 				gene_regions=gene_df[gene_df['gene_id'] == gene_id],
	# 				embedding_window=(n_upstream, n_downstream),
	# 				cnv_df=cnv_df[cnv_columns],
	# 				mode=mode
	# 			)

	# 			# accumulate all barcode data for this gene
	# 			for j, (cnv_barcode, cnv_gene_id, cnv_embedding) in \
	# 				enumerate(cnv_embedder):
	# 				assert gene_barcode_list[j] == cnv_barcode, "{} != {}".format(
	# 					gene_barcode_list[j], cnv_barcode
	# 				)
	# 				barcode_embeddings.append(cnv_embedding)
				
	# 			# repreat genomic embedding for each barcode and concat them
	# 			genomic_embedding_tile = tile(
	# 				genomic_embedding, reps=(len(barcode_embeddings),1,1)
	# 			)
	# 			barcode_embedding_tile = vstack(barcode_embeddings).reshape(
	# 					(len(barcode_embeddings), *cnv_embedding.shape)
	# 				)
	# 			yield (
	# 				'all_barcodes',
	# 				gene_id,
	# 				np_concat(
	# 					[genomic_embedding_tile, barcode_embedding_tile],
	# 					axis=1
	# 				)
	# 			)
	# 			barcode_embeddings = list()
				
	# 		case 'single_gene_barcode':
	# 			# get barcodes selected for this gene
	# 			gene_barcode_list = gene_to_barcodes[gene_id]
	# 			cnv_columns = ['seq', 'start', 'end']
	# 			cnv_columns.extend(gene_barcode_list)
	# 			if verbose:
	# 				print('[embed]: barcode_list', gene_barcode_list)

	# 			cnv_embedder = embed_cnv(
	# 				gene_regions=gene_df[gene_df['gene_id'] == gene_id],
	# 				embedding_window=(n_upstream, n_downstream),
	# 				cnv_df=cnv_df[cnv_columns],
	# 				mode=mode
	# 			)

	# 			# iterate through all barcodes for this gene and yield single
	# 			#  embeddings
	# 			for j, (cnv_barcode, cnv_gene_id, cnv_embedding) in \
	# 				enumerate(cnv_embedder):
	# 				assert gene_id == cnv_gene_id and \
	# 					gene_barcode_list[j] == cnv_barcode, \
	# 					"{}: gene {} ?= (CNV) {} for barcode {} ?= (CNV) {}".format(
	# 						j, gene_id, cnv_gene_id, gene_barcode_list[j], cnv_barcode
	# 					)
	# 				yield (
	# 					cnv_barcode,
	# 					gene_id,
	# 					vstack([genomic_embedding, cnv_embedding])
	# 				)
				
	# 		case 'gene_concat':
	# 			cnv_barcode, cnv_gene_id, cnv_embedding = next(cnv_embedder)
	# 			assert gene_id == cnv_gene_id
	# 			barcode_embeddings.append(cnv_embedding)

	# if mode == 'gene_concat':
	# 	yield (
	# 		cnv_barcode,
	# 		'all_genes',
	# 		vstack([
	# 			hstack(genomic_embeddings),
	# 			hstack(barcode_embeddings)
	# 		])
	# 	)

	# 	barcode_embeddings = []

	# 	# stop iterating here, if there was only one barcode
	# 	if len(uniq_barcodes) == 1:
	# 		return

	# 	# generate all barcode embeddings for remaining barcodes
	# 	cnv_barcode, cnv_gene_id, cnv_embedding = next(cnv_embedder)
	# 	for next_cnv_barcode, next_cnv_gene_id, next_cnv_embedding in \
	# 		cnv_embedder:
	# 		if cnv_barcode != next_cnv_barcode:
	# 			barcode_embeddings.append(cnv_embedding)
	# 			yield (
	# 				cnv_barcode,
	# 				'all_genes',
	# 				vstack([
	# 					hstack(genomic_embeddings),
	# 					hstack(barcode_embeddings)
	# 				])
	# 			)
	# 			barcode_embeddings = []
	# 		else:
	# 			barcode_embeddings.append(cnv_embedding)
			
	# 		cnv_barcode = next_cnv_barcode	
	# 		cnv_gene_id = next_cnv_gene_id
	# 		cnv_embedding = next_cnv_embedding	

	# 	# handle fence post
	# 	barcode_embeddings.append(cnv_embedding)
	# 	yield (
	# 		cnv_barcode,
	# 		'all_genes',
	# 		vstack([
	# 			hstack(genomic_embeddings),
	# 			hstack(barcode_embeddings)
	# 		])
	# 	)


class Embedder(object):
	"""
	Main wrapper function. Generates embeddings from following files:
	* fasta reference genome sequence
	* overlap data of ATAC-seq peaks and genes
	* CNV results from from EpiAneufinder (add column name for index in 1st row)
	* gtf genome annotation (optional, only if CDS, promoter, ... data wanted)

	Embeddings will be returned in genomic gene order (1st gene on chr1,
	2nd gene on chr1, ... last gene on chrY) and alphabetical barcode order.

	fasta_path : str path to reference genome .fasta file
	atac_path : str path to overlap data from ATAC-seq peaks and genes (.tsv)
	cnv_path : str path to results from from EpiAneufinder (.tsv file)
	barcode_to_genes : Dict[str, List[str]] of ENSEMBL gene id lists (values)
		for different barcodes (keys).
	gene_set : Set[str] or None specifiying subset of ENSEMBL gene ids to use
	barcode_set : Set[str] or None specifying subset of cell barcodes to use
	mode : str one of ['gene_concat', 'barcode_channel', 'single_gene_barcode']
		to specify how to combine and return embeddings. Output shapes:
		* 'gene_concat': (n_coding_rows, embedding_len * n_genes) per barcode
		* 'barcode_channel': (n_barcodes, n_coding_rows, embedding_len) per gene
		* 'single_gene_barcode': (n_coding_rows, embedding_len) per gene - barc.
		Default: 'gene_concat'
	pad_dna : bool if pad embedding positions after gene end. Default: True
	n_upstream : int of embedding length upstream of gene start. Default: 2000
	n_downstream : int of embedding len. downstram of gene start. Default: 8000
	gtf_path : None or str path to .gtf annotation file

	returns : Tuple[str,str,ndarray] of cell/barcode id, gene id and embedding
		as numpy.ndarray. Note depending on mode the value of gene_id or 
		barcode_id might be 'all barcodes' (in case of 'barcode_channel') or 
		'all genes' ('gene_concat'). This is due to the fact, that in these
		modes all barcodes or genes are represented in the returned embedding.
	
	TODO: update documentation
	"""

	# TODO: use pyranges to parse gene entries
	# * filter genes not present in gtf -> raise exception
	# * quick fix: use gene positions and overlap file from gaja
	# import pyranges as pr
	# gtf = pr.read_gtf('data/reference/Homo_sapiens.GRCh38.113.gtf.gz')
	# import sys
	# sys.getsizeof(gtf)
	# gtf[(gtf.Feature == 'gene') & (gtf.Chromosome.isin(['1', '21']))].df.loc[0]
	# pr.get_sequence(gtf[(gtf.Feature == 'gene') & (gtf.gene_id == 'ENSG00000142611')], 'data/reference/GRCh38.d1.vd1.fa')
	# use from pympler import asizeof for memory checks

	# TODO:
	# 1. read gft/genes file and filter for genes in gene_set
	# 2. compute embedding start and end from gene start / end
	# 3. switch start and end columns with emb start and end
	# 4. convert to PyRanges
	# 5. get sequences from fasta
	# 6. create data table with all wanted gene-barcode combinations
	# 7. iterate over gene_id, barcode pairs
	#   7.1. get ATAC per barcode
	#	7.2. get CNVs per barcode

	def __init__(self, fasta_path: Path, gtf_path: Path, atac_path: Path, 
				cnv_path: Path, mode='single_gene_barcode',
				barcode_to_genes: Union[Dict[str, List[str]], None]=None,
				barcode_set: Union[Set[str], None]=None,
				gene_set: Union[Set[str], None]=None, verbose=False,
				pad_dna=True, n_upstream=2000, n_downstream=8000, dtype=uint8
				):
		
		# self.super().__init__()
		assert fasta_path.is_file(), 'FASTA not found: {}'.format(fasta_path)
		assert gtf_path.is_file(), 'GTF not found: {}'.format(gtf_path)
		assert atac_path.is_file(), 'Overlaps not found: {}'.format(atac_path)
		assert cnv_path.is_file(), 'CNV file not found: {}'.format(cnv_path)
		assert mode == 'single_gene_barcode', \
			'[Embedder]: Only supporting "single_gene_barcode" mode for now!'
		
		self.mode = mode
		self.embedding_size = (n_upstream, n_downstream)
		self.fasta_path = fasta_path
		self.gtf_path = gtf_path
		self.atac_path = atac_path
		self.cnv_path = cnv_path
		self.dtype = dtype

		# === GTF ANNOTATION ===
		# TODO: use pyranges to read annotation from gtf: 
		# for now use 'data/gene_positions_and_overlaps/gene_positions.csv'
		gene_df = read_csv(gtf_path)
		gene_df = gene_df.rename(columns={
			'seqnames': 'Chromosome', 'start': 'Gene_Start', 'end': 'Gene_End'
		})
		gene_df = gene_df[gene_df['Chromosome'].isin(standard_chromosomes)]
		uniq_gene_ids = set(gene_df['gene_id'])

		# apply subsetting by gene_id
		if gene_set is not None:
			uniq_gene_overlap = uniq_gene_ids.intersection(gene_set)
			if len(gene_set) - len(uniq_gene_overlap) > 0:
				print('[Embedder]: No GTF annot. for {} genes from gene_set'.format(
					len(gene_set) - len(uniq_gene_overlap)
				))
			if verbose:
				print(','.join(
					[gid for gid in gene_set if gid not in uniq_gene_overlap]
				))
			uniq_gene_ids = uniq_gene_overlap
		if barcode_to_genes is not None:
			iter_gene_set = {
				gene for vals in barcode_to_genes.values() for gene in vals
			}
			uniq_gene_overlap = uniq_gene_ids.intersection(iter_gene_set)
			if len(iter_gene_set) - len(uniq_gene_overlap) > 0:
				print('[Embedder]: No GTF annot. for {} genes from barcode_to_genes'\
					.format(len(iter_gene_set) - len(uniq_gene_overlap))
				)
			if verbose:
				print(','.join(
					[gid for gid in iter_gene_set if gid not in uniq_gene_overlap]
				))
			uniq_gene_ids = uniq_gene_overlap

		gene_df = gene_df[gene_df['gene_id'].isin(uniq_gene_ids)]

		# calculate embedding starts
		gene_df['Start'] = gene_df['Gene_Start'] - n_upstream
		gene_df['End'] = gene_df['Gene_Start'] + n_downstream

		# convert to pyranges
		gene_pr = pr.PyRanges(gene_df)
		gene_pr.Sequence = pr.get_sequence(gene_pr, fasta_path)
		self.gene_pr = gene_pr
		if verbose:
			print('[Embedder]:\n', self.gene_pr)

		# ==== OPEN CHROMATIN ====
		# load open chromatin peaks
		atac_df = read_csv(atac_path, sep='\t')
		atac_df = atac_df.rename(columns={
			'Start_peak': 'Start', 'End_peak': 'End'
		})
		self.atac_pr = pr.PyRanges(atac_df)
		
		# ==== CNV ====
		# load EpiAneufinder result table 
		cnv_df = read_csv(cnv_path, sep=' ')
		assert all(cnv_df.columns[1:4] == ['seq', 'start', 'end']),\
			'Column name Mismatch, please prepend \'idx \' to the first line.'
		
		# define unique barcodes to use for embedding computation
		uniq_barcodes = set(cnv_df.columns[4:])
		if barcode_set is not None:
			barcode_diff = barcode_set.difference(uniq_barcodes)
			if len(barcode_diff) > 0:
				warn('[Embedder]: No CNV data for {}'.format(','.join(barcode_diff)))
			uniq_barcodes = uniq_barcodes.intersection(barcode_set)

		# remove not needed barcodes
		cnv_df = cnv_df[
			cnv_df.columns[
				cnv_df.columns.isin(uniq_barcodes.union({'seq', 'start', 'end'}))
			]
		]

		# cnv_df = cnv_df.sort_values(by=['seq', 'start', 'end'])
		cnv_df = cnv_df.astype({c: 'u1' for c in cnv_df.columns[4:]})
		cnv_df['seq'] = Series(map(lambda x: x.replace('chr', ''), cnv_df['seq']))
		cnv_df = cnv_df.rename(columns={
			'seq': 'Chromosome', 'start': 'Start', 'end': 'End'
		})
		# print('[Embedder]:', cnv_df)
		# cnv_df = cnv_df.drop('idx', axis=1)
		cnv_df = pr.PyRanges(cnv_df)
		#cnv_df = cnv_df.overlap(gene_df) # only retain overlaps with relevant genes
		self.cnv_pr = self.gene_pr.join(cnv_df) # do inner join
		# TODO: test overlaps at 100k points
		# gc_join = gene_pr.join(cnv_pr)
		# gc_join.df.apply(lambda r: int(str(r['Start'])[0]) < int(str(r['End'])[0]), axis = 1)
		if verbose:
			print('[Embedder]:\n', self.cnv_pr)

		# ==== Iteration Mapping ====
		# create list of tuples based on barcode_to_genes dict
		gene_barcode_pairs = list()
		n_embeddings = 0
		if barcode_to_genes is not None:
			n_embeddings = len([
				g for gl in barcode_to_genes.values() for g in gl 
				if g in uniq_gene_ids
			])
			print('[Embedder]: Iterating over custom barcode to genes mapping')
			iter_barcode_set = set(barcode_to_genes.keys())
			uniq_barcodes = uniq_barcodes.intersection(iter_barcode_set)

			# sort uniq barcodes for alphabetical iteration order
			uniq_barcodes = sorted(list(uniq_barcodes))

			# create a mapping from gene to barcode to serve for iteration later
			gene_to_barcodes = {gene: list() for gene in uniq_gene_ids}
			for cnv_barcode in uniq_barcodes:
				for gene in barcode_to_genes[cnv_barcode]:
					if gene in uniq_gene_ids:
						gene_to_barcodes[gene].append(cnv_barcode)

			gene_barcode_pairs = [
				(g, b) for g in self.gene_pr.df['gene_id'] for b in gene_to_barcodes[g]
			]
			
		else:
			print('[Embedder]: Iterating over all possible barcode-gene combinations')
			n_embeddings = len(uniq_barcodes) * len(uniq_gene_ids)

			# sort uniq barcodes for alphabetical iteration order
			uniq_barcodes = sorted(list(uniq_barcodes))

			gene_barcode_pairs = [
				(g, b) for g in self.gene_pr.df['gene_id'] for b in uniq_barcodes
			]

		if len(uniq_barcodes) == 0:
			raise RuntimeError(
				'No barcode data available!\n' +
				'Please make sure the barcodes you want to use are present in the' +
				' epiAneufinder results in order to have CNV data.'
			)
		if len(uniq_gene_ids) == 0:
			raise RuntimeError(
				'No ENSEMBL gene ids in overlap!\n' +
				'Please make sure there is both ATAC and CNV data for the genes ' +
				'you want to process.'
			)

		assert len(gene_barcode_pairs) == n_embeddings, \
			'Iteration mapping failed! Incorrect number of embeddings'
		self.n_embeddings = n_embeddings

		print('[Embedder]: Computing {} Embeddings with mode: "{}"'.format(
			n_embeddings, mode
		))
		print('[Embedder]: Using {} barcodes'.format(len(uniq_barcodes)))
		if verbose:
			print('[Embedder]:', ','.join(uniq_barcodes))
		print('[Embedder]: Using {} genes'.format(len(uniq_gene_ids)))
		if verbose:
			print('[Embedder]:', ','.join(uniq_gene_ids))

		self.uniq_barcodes = uniq_barcodes
		self.uniq_gene_ids = uniq_gene_ids

		self.embegging_iterator = iter(gene_barcode_pairs)
		self.pbar = tqdm(
			total=n_embeddings,
			ncols=120,
			desc='[Embedder]: Computing embeddings'
		)

		self.prev_gene_id = None
		self.dna_embedding = None

	def __len__(self):
		return self.n_embeddings

	def __iter__(self):
		return self
    
	def __next__(self):
		try:
			gene_id, barcode = next(self.embegging_iterator)
		except StopIteration:
			self.pbar.close()
			raise StopIteration()
		
		# dna sequence embedding
		if gene_id != self.prev_gene_id:
			print('[Embedder]: get_dna_embedding() for {}'.format(gene_id))
			self.dna_embedding = get_dna_embedding(
				self.gene_pr[self.gene_pr.gene_id == gene_id],
				fasta_path=self.fasta_path,
				dtype=self.dtype
			)
			self.prev_gene_id = gene_id

		# get open chromatin emebdding
		atac_embedding = get_atac_embedding(
			self.atac_pr[
				(self.atac_pr.gene_id == gene_id) &
				(self.atac_pr.barcode == barcode)
			].overlap(self.gene_pr),
			embedding_window=self.embedding_size,
			dtype=self.dtype
		)

		# get CNV embedding
		cnv_embedding = get_cnv_embedding(
			region=self.gene_pr[self.gene_pr.gene_id == gene_id],
			cnv_pr=self.cnv_pr[self.cnv_pr.gene_id == gene_id][[barcode]],
			barcode=barcode,
			dtype=self.dtype
		)

		self.pbar.update(1)

		return (
			barcode,
			gene_id,
			vstack(
				[self.dna_embedding, atac_embedding, cnv_embedding],
				dtype=self.dtype
			)
		)
