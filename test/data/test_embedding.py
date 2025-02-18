from numpy import array, array_equal, zeros, uint8
from pandas import DataFrame, read_csv, concat
from data.embedding import (
	encode_dna_seq,
	encode_open_chromatin,
	encode_cnv_status,
	embed_dna,
	embed_atac,
	embed_cnv,
	embed
)
from src.util import relative_idx
from src.data import autosomes, allosomes


def test_encode_dna_seq():
	# short test case
	dna_seq = "CGATGCTGTGATGC"
	expected_result = array([
		# one hot encoding
		[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
		[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
		[0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
		[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
	], dtype='u1')
	print(encode_dna_seq(dna=dna_seq))
	assert array_equal(encode_dna_seq(dna=dna_seq), expected_result)

	# simple padding test case
	dna_seq = "NNNTGCTGTGANNN"
	expected_result = array([
		# one hot encoding
		[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
		[1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
		[1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
		[1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1]
	], dtype='u1')
	assert array_equal(encode_dna_seq(dna=dna_seq), expected_result)

	# long test case
	dna_seq = "TTCTCTTCGGAGCCAGGAACCAGCTCTTCCAGTGCTGGGGTTTT" + \
		"CACCGAGGACGACATGCTGAAGCCAC"
	# base_idx = [[i for i in range(len(dna_seq)) if dna_seq[i] == base] for base in "ACGT"]
	# expected_result = array(
	#       [
	#             [1 if i in base_i else 0 for i in range(len(dna_seq))] \
	#                   for base_i in base_idx
	#       ]
	# )
	expected_result = array([
		# one hot encoding
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
		[0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
		[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
		[1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	], dtype='u1')
	assert array_equal(encode_dna_seq(dna=dna_seq), expected_result)


def test_encode_open_chromatin():
	# TODO: account for 0- to 1-based difference and end-inclusive intervals
	#  use correct=-1 for relative_idx
	upsteam = 2000
	downsteam = 8000

	# def relative_idx(i, emb_start, emb_end, correct=0):
	# 	if emb_end < i:
	# 		i = emb_end
	# 	assert i >= emb_start and i <= emb_end
	# 	return i - emb_start + correct
	
	# simple one peak case
	embed_interval = [0, 20]
	peak_intervals = [(5, 10)]
	arr_expected = zeros((1, embed_interval[1] - embed_interval[0]), dtype='u1')
	for start, end in peak_intervals:
		rel_start = relative_idx(start, embed_interval)
		rel_end = relative_idx(end, embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)

	# simple two peak case
	embed_interval = [0, 20]
	peak_intervals = [(5, 10), (13, 17)]
	arr_expected = zeros((1, embed_interval[1] - embed_interval[0]), dtype='u1')
	for start, end in peak_intervals:
		rel_start = relative_idx(start, embed_interval)
		rel_end = relative_idx(end, embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)

	# simple four peak case
	embed_interval = [0, 20]
	peak_intervals = [(1, 3), (5, 10), (13, 17), (19, 22)]
	arr_expected = zeros((1, embed_interval[1] - embed_interval[0]), dtype='u1')
	for start, end in peak_intervals:
		rel_start = relative_idx(start, embed_interval)
		rel_end = relative_idx(end, embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)

	# single peak case: ENSG00000290825
	embed_interval = [9121 - upsteam, 9121 + downsteam]
	peak_intervals = [(9855, 10676)]
	arr_expected = zeros((1, embed_interval[1] - embed_interval[0]), dtype='u1')
	for start, end in peak_intervals:
		rel_start = relative_idx(start, embed_interval)
		rel_end = relative_idx(end, embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)

	# multiple peak case: ENSG00000228794
	embed_interval = [823138 - upsteam, 823138 + downsteam]
	peak_intervals = [
		(822918, 823607), (825354, 826159), (827067, 827928), (835242, 836034),
		(838089, 838689), (851462, 852319), (854606, 855500), (869498, 870402)
	]
	arr_expected = zeros((1, embed_interval[1] - embed_interval[0]), dtype='u1')
	for start, end in peak_intervals:
		if start > embed_interval[1]:
			continue 
		rel_start = relative_idx(start, embed_interval)
		rel_end = relative_idx(end, embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)


def test_encode_cnv_status():
	# missing data test
	embed_interval = [0, 20]
	cnv_interval_status = [((),)]
	arr_expected = zeros((2, embed_interval[1]))
	assert array_equal(
		encode_cnv_status(embed_interval, cnv_interval_status),
		arr_expected
	)

	# normal CNV test, single interval
	embed_interval = [0, 20]
	cnv_interval_status = [((7, 15), 1)]
	arr_expected = zeros((2, embed_interval[1]))
	assert array_equal(
		encode_cnv_status(embed_interval, cnv_interval_status),
		arr_expected
	)

	# loss CNV test, single interval
	embed_interval = [0, 20]
	cnv_interval_status = [((7, 15), 0)]
	arr_expected = zeros((2, embed_interval[1]))
	arr_expected[0,7:15] = 1
	assert array_equal(
		encode_cnv_status(embed_interval, cnv_interval_status),
		arr_expected
	)

	# gain CNV test, single interval
	embed_interval = [0, 20]
	cnv_interval_status = [((3, 11), 2)]
	arr_expected = zeros((2, embed_interval[1]))
	arr_expected[1,3:11] = 1
	assert array_equal(
		encode_cnv_status(embed_interval, cnv_interval_status),
		arr_expected
	)

	# mixed CNV test, multi interval
	embed_interval = [0, 20]
	cnv_interval_status = [((3, 7), 2), ((10, 13), 1), ((18, 20), 0)]
	arr_expected = zeros((2, embed_interval[1]))
	arr_expected[0,18:20] = 1
	arr_expected[1,3:7] = 1
	assert array_equal(
		encode_cnv_status(embed_interval, cnv_interval_status),
		arr_expected
	)


def test_embed_dna():
	fasta_path='data/GRCh38.d1.vd1.fa'
	test_regions = DataFrame({
		'Chromosome': ['1'] * 2,
		'Start_gene': [9121, 26894],
		'Eng_gene': [584945, 829989],
		'gene_id': ['foo', 'bar']
	})
	embed_size = (2000, 8000)
	dna_embedder = embed_dna(
		gene_regions=test_regions,
		embedding_window=embed_size,
		fasta_path=fasta_path
	)

	for _, (c, s, e, _ ) in test_regions.iterrows():
		chrom, embd_start, embd_end, dna_embedding = next(dna_embedder)
		# assert c == chrom and embd_start == s - embed_size[0] and embd_end == e + embed_size[1]
		assert dna_embedding.shape == (4, sum(embed_size))


def test_embed_atac():
	atac_path='data/overlap_genes_peaks.tsv'
	test_regions = DataFrame({
		'Chromosome': ['1'] * 2,
		'Start_gene': [9121, 26894],
		'Eng_gene': [584945, 829989],
		'gene_id': ['foo', 'bar']
	})
	embed_size = (2000, 8000)

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

	atac_embedder = embed_atac(
		gene_regions=test_regions,
		embedding_window=embed_size,
		atac_df= atac_df
	)

	for _, (c, s, e, _ ) in test_regions.iterrows():
		chrom, embd_start, embd_end, atac_embedding = next(atac_embedder)
		# assert c == chrom and embd_start == s - embed_size[0] and embd_end == e + embed_size[1]
		assert atac_embedding.shape == (1, sum(embed_size))


def test_embed_cnv():
	epianeu_path='data/epiAneuFinder_results.tsv'
	test_regions = DataFrame({
		'Chromosome': ['1'] * 2,
		'Start_gene': [9121, 26894],
		'Eng_gene': [584945, 829989],
		'gene_id': ['foo', 'bar']
	})
	embed_size = (2000, 8000)
	cnv_embedder = embed_cnv(
		gene_regions=test_regions,
		embedding_window=embed_size,
		cnv_path=epianeu_path
	)
	for _, (c, s, e, _ ) in test_regions.iterrows():
		print(c, ':', str(s), '-', str(e))
		barcode, gene_id, cnv_embedding = next(cnv_embedder)
		assert cnv_embedding.shape == (2, sum(embed_size)), "{} wrong CNV shape".format(barcode)


def test_embed_main():
	# gft_path='data/Homo_sapiens.GRCh38.113.gtf.gz'
	fasta_path='data/GRCh38.d1.vd1.fa'
	overlap_path='data/overlap_genes_peaks.tsv'
	epianeu_path='data/epiAneuFinder_results.tsv'
	test_genes = [
		'ENSG00000000971',
		'ENSG00000002587',
		'ENSG00000002745'
	]
	test_gene_set = set(test_genes)
	test_barcodes = [
		'cell-GCGCAATGTTGCGGAT-3',
		'cell-CTAGTGAGTCACCTAT-3',
		'cell-AATCATGTCGATCAGT-1'
	]
	test_barcode_set = set(test_barcodes)

	# single_gene_barcode test case
	embedder = embed(
		fasta_path,
		overlap_path,
		epianeu_path,
		gene_set=test_gene_set,
		barcode_set=test_barcode_set,
		mode='single_gene_barcode'
	)
	i = 0
	for gene in test_genes:
		for barcode in test_barcodes:
			barcode_id, gene_id, embedding = next(embedder)
			print(i,
				"\nexpected:", barcode, ' and ', gene, 
				'\nactual',barcode_id, gene_id
			)
			assert gene_id == gene
			assert barcode_id == barcode
			assert embedding.shape == (7, 10_000), \
				"Wrong shape: {},{}".format(*embedding.shape)
			i += 1

	assert i == len(test_genes) * len(test_barcodes)

	# TODO: single_gene_barcode for single barcode and multiple genes

	# TODO single_gene_barcode for multiple barcodes and single gene

	# gene concat case
	embedder = embed(
		fasta_path,
		overlap_path,
		epianeu_path,
		gene_set=test_gene_set,
		barcode_set=test_barcode_set,
		mode='gene_concat'
	)
	for i in range(3):
		barcode_id, _, embedding = next(embedder)
		assert barcode_id == test_barcodes[i]
		assert embedding.shape == (7, 3 * 10_000),\
			"Wrong shape: {},{}".format(*embedding.shape)

	# barcode channel case
	embedder = embed(
		fasta_path,
		overlap_path,
		epianeu_path,
		gene_set=test_gene_set,
		barcode_set=test_barcode_set,
		mode='barcode_channel'
	)
	for i in range(3):
		_, gene_id, embedding = next(embedder)
		assert gene_id == test_genes[i]
		assert embedding.shape == (3, 7, 10_000),\
			"Wrong shape: {},{}".format(*embedding.shape)
