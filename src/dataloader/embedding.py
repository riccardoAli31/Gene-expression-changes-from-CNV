#!/bin/python

"""
Script for creating embeddings from DNA regions.
See https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x/figures/5 
"""

import numpy as np
from numpy import ndarray	
from typing import List, Tuple
import regex as re

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

def genomic_regions_generator(gene_ids: List[str], path_to_gft='data/Homo_sapiens.GRCh38.113.gtf.gz'):
	"""
	Generator that extracts genomic regions for genes of interest.

	gene_ids : List of str containting ENSEMBL gene identifiers (e.g. ENSG00000142611)
				This list should be sorted by genomic position.
	path_to_gft : str of the path to .gft file
	"""

	# sample gft line
	# 1	ensembl_havana	gene	3069168	3438621	.	+	.	gene_id "ENSG00000142611"; gene_version "17"; gene_name "PRDM16"; gene_source "ensembl_havana"; gene_biotype "protein_coding";
	
	gene_id_iterator = iter(gene_ids)

	with open(path_to_gft, 'r') as gtf_file:
		gene_id = next(gene_id_iterator)
		for line in gtf_file:
			if re.match(gene_id, line):
				line = line.strip().split('\t')
				if line[2] == "gene":
					# assert gene_id match TODO: remove this
					assert line[8].split(';')[0].split('\"')[-2] == gene_id
					# TODO assert gene_biotype "protein_coding"?

					chrom, _, _, start, end, _, strand = line[:6]
					yield (int(chrom), int(start), int(end), strand, gene_id)
					gene_id = next(gene_id_iterator)

def fasta_generator(fasta_file='data/GRCh38.d1.vd1.fa.tar.gz'):
	"""
	Generator for chromosome fasta entries in the order saved in file.
	Assert that the chromosomes are saved in increasing order, 
	 starting with Autosomes.
	"""
	
	seq_id, seq = None, None
	with open(fasta_file, 'r') as fa_file:
		for line in fa_file:
			if line.startswith('>'):
				if seq_id is not None:
					yield (seq_id, seq.upper())
				seq_id = line.strip()
			else:
				seq += line.strip()


def one_hot_dna(dna_string: str, alphabet=set("ACGT")) -> ndarray:
	"""
	Function to produce one-hot encodings for DNA

	dna_string : str of DNA sequence with posibly N-padding
	alphabet : set of bases to encode. If you want to encode the 
			N padding as a separate row, please adapt this here.  
	"""
	
	dna_string = dna_string.upper()
	ignore_N = 'N' not in alphabet

	one_hot_dna = np.zeros((len(alphabet), len(dna_string)), dtype='u1')
	for i, aa in enumerate(dna_string):
		if ignore_N and aa == 'N':
			continue
		one_hot_dna[alphabet.index(aa), i] = 1

	return one_hot_dna


def encode_CDS_structure(CDS_start_stop: List[Tuple[int, int]], seq_len: int) -> ndarray:
	# TODO
	CDS_structure = np.zeros(seq_len, dtype='u1')
	for start, stop in CDS_start_stop:
		CDS_structure[start:stop] = 1

	return CDS_structure


def encode_reading_frame(CDS_start_stop: List[Tuple[int, int]],seq_len: int):
	# TODO compute reading frame from CDS structure
	codon_size = 3
	reading_frame = np.zeros(seq_len, dtype='u1')
	overhead = 0
	for start, stop in CDS_start_stop:
		reading_frame[list(start + (codon_size - overhead), stop, codon_size)] = 1
		overhead = (stop - start + overhead) % 3

	return reading_frame


def combine_embedding(rows: List[ndarray], mode: str):
	"""
	Wrapper that stacks different embedding parts (e.g. DNA one-hot, open chromatin, etc.)
	
	"""
	assert all(map(rows, lambda x: x.shape[1] == rows[0].shape[1]))

	embedding = np.vstack(rows)

	# TODO create embeddings in different modes: gene concatenation per sample or sample channels per gene
	match mode:
		case 'gene_concat':
			return embedding
		case 'sample_channel':
			pass


def embed():
	"""
	Main wrapper function.
	"""
	pass
