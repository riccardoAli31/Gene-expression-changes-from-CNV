import pytest
import numpy as np
from src.dataloader.embedding import *

include_file_reading_generators = False

def test_relative_idx():
	idx = 42
	interval = (13, 50)
	assert relative_idx(idx, interval) == 29

	idx = 42
	interval = (13, 50)
	assert relative_idx(idx, interval, correct=-1) == 28

	idx = 5
	interval = (13, 50)
	with pytest.raises(AssertionError):
		relative_idx(idx, interval, clip_start=False)


def test_interval_overlap():
	interval_1 = (0, 0)
	interval_2 = (2, 3)
	with pytest.raises(AssertionError):
		interval_overlap(interval_1, interval_2)
	
	interval_1 = (0, -3)
	interval_2 = (2, 3)
	with pytest.raises(AssertionError):
		interval_overlap(interval_1, interval_2)

	interval_1 = (10, 20)
	interval_2 = (22, 30)
	assert interval_overlap(interval_1, interval_2) == None

	interval_1 = (10, 20)
	interval_2 = (2, 3)
	assert interval_overlap(interval_1, interval_2) == None

	interval_1 = (10, 20)
	interval_2 = (15, 30)
	assert interval_overlap(interval_1, interval_2) == (15, 20)

	interval_1 = (20, 42)
	interval_2 = (15, 30)
	assert interval_overlap(interval_1, interval_2) == (20, 30)


def test_dna_padding():
	dna = 'CTCTAGCTAGCTAGCTATCG'
	assert dna_padding(dna, 15) == 'CTCTAGCTAGCTAGCNNNNN'


def test_encode_dna_seq():
	# short test case
	dna_seq = "CGATGCTGTGATGC"
	expected_result = np.array([
		# one hot encoding
		[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
		[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
		[0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
		[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
	], dtype='u1')
	print(encode_dna_seq(dna=dna_seq))
	assert np.array_equal(encode_dna_seq(dna=dna_seq), expected_result)

	# simple padding test case
	dna_seq = "NNNTGCTGTGANNN"
	expected_result = np.array([
		# one hot encoding
		[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
		[1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
		[1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
		[1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1]
	], dtype='u1')
	assert np.array_equal(encode_dna_seq(dna=dna_seq), expected_result)

	# long test case
	dna_seq = "TTCTCTTCGGAGCCAGGAACCAGCTCTTCCAGTGCTGGGGTTTT" + \
		"CACCGAGGACGACATGCTGAAGCCAC"
	# base_idx = [[i for i in range(len(dna_seq)) if dna_seq[i] == base] for base in "ACGT"]
	# expected_result = np.array(
	#       [
	#             [1 if i in base_i else 0 for i in range(len(dna_seq))] \
	#                   for base_i in base_idx
	#       ]
	# )
	expected_result = np.array([
		# one hot encoding
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
		[0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
		[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
		[1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	], dtype='u1')
	assert np.array_equal(encode_dna_seq(dna=dna_seq), expected_result)


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
	arr_expected = np.zeros((1, embed_interval[1] - embed_interval[0]), dtype='u1')
	for start, end in peak_intervals:
		rel_start = relative_idx(start, embed_interval)
		rel_end = relative_idx(end, embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert np.array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)

	# simple two peak case
	embed_interval = [0, 20]
	peak_intervals = [(5, 10), (13, 17)]
	arr_expected = np.zeros((1, embed_interval[1] - embed_interval[0]), dtype='u1')
	for start, end in peak_intervals:
		rel_start = relative_idx(start, embed_interval)
		rel_end = relative_idx(end, embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert np.array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)

	# simple four peak case
	embed_interval = [0, 20]
	peak_intervals = [(1, 3), (5, 10), (13, 17), (19, 22)]
	arr_expected = np.zeros((1, embed_interval[1] - embed_interval[0]), dtype='u1')
	for start, end in peak_intervals:
		rel_start = relative_idx(start, embed_interval)
		rel_end = relative_idx(end, embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert np.array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)

	# single peak case: ENSG00000290825
	embed_interval = [9121 - upsteam, 9121 + downsteam]
	peak_intervals = [(9855, 10676)]
	arr_expected = np.zeros((1, embed_interval[1] - embed_interval[0]), dtype='u1')
	for start, end in peak_intervals:
		rel_start = relative_idx(start, embed_interval)
		rel_end = relative_idx(end, embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert np.array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)

	# multiple peak case: ENSG00000228794
	embed_interval = [823138 - upsteam, 823138 + downsteam]
	peak_intervals = [
		(822918, 823607), (825354, 826159), (827067, 827928), (835242, 836034),
		(838089, 838689), (851462, 852319), (854606, 855500), (869498, 870402)
	]
	arr_expected = np.zeros((1, embed_interval[1] - embed_interval[0]), dtype='u1')
	for start, end in peak_intervals:
		if start > embed_interval[1]:
			continue 
		rel_start = relative_idx(start, embed_interval)
		rel_end = relative_idx(end, embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert np.array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)


def test_extract_cnv_overlaps():
	starts = [300_001, 500_001, 600_001, 700_001, 800_001]
	ends = [400_000, 600_000, 700_000, 800_000, 900_000]
	barcode_status = {
		'cell-GCGCAATGTTGCGGAT-3': [1, 1, 1, 1, 1],
		'cell-CTAGTGAGTCACCTAT-3': [1, 1, 0, 0, 0],
		'cell-AATCATGTCGATCAGT-1': [1, 1, 1, 1, 1]
	}
	df = pd.concat([
		pd.DataFrame({
			'seq': ['1'] * len(starts),
			'start': starts,
			'end': ends,
		}), 
		pd.DataFrame(barcode_status)],
		axis=1
	)
	print(df)
	regions = [
		('1', 9_121, 26_894),
		('1', 584_945, 829_989)
	]
	barcode = 'cell-CTAGTGAGTCACCTAT-3'
	expected_results = [
		[((),)],
		[
			((584_945, 600_000), 1),
			((600_001, 700_000), 0),
			((700_001, 800_000), 0),
			((800_001, 829_989), 0)
		]
	]

	for region, expected_result in zip(regions, expected_results):
		cnv_overlaps = extract_cnv_overlaps(df, barcode, region)
		assert cnv_overlaps == expected_result




# def test_encode_CNV_regions():
#       # TODO
#       pass

# def test_encode_CDS_structure():
#       # TODO
#       exon_start_stop = [(2, 10)]
#       expected_result = np.array([
#             # one hot encoding
#             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#             [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
#             [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
#             # reading frame
#             [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
#             # exon structure
#             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
#       ])
#       # TODO two exon test case
#       """ dna_string = "CGATGCTGTGATGC"
#       exon_start_stop = [(0, 16), (23, 33)]
#       expected_result = np.array([
#             []
#       ]) """
#       # TODO real gene test case at 4kb window
#       """ dna_string = "CGATGCTGTGATGC"
#       exon_start_stop = [(0, 16), (23, 33)]
#       expected_result = np.array([
#             []
#       ]) """
#       pass

def test_generate_fasta_entries():
	if include_file_reading_generators:
		standard_chroms = [
			'1', '2', '3', '4', '5', '6', '7', 
			'8', '9', '10', '11', '12', '13', 
			'14', '15', '16', '17', '18', '19',
			'20', '21', 'X', 'Y'
		]
		base_counts = {}
		for chrom, seq in generate_fasta_entries():
			assert chrom in standard_chroms
			assert len(seq) > 0
			seq_counts = {base: seq.count(base) for base in set(seq)}
			base_counts = {
				base_counts[base]: base_counts[base] + count
				if base in base_counts else count 
				for base, count in seq_counts
			}

		all_chroms = [
			chrom for chrom, _ in generate_fasta_entries(only_standard_chrom=False)
		]
		assert len(standard_chroms) <= len(all_chroms)


def test_generate_genomic_regions():
	if include_file_reading_generators:
		first_seven_genes = [
			'ENSG00000142611', 'ENSG00000284616', 'ENSG00000157911',
			'ENSG00000260972', 'ENSG00000224340', 'ENSG00000229280',
			'ENSG00000142655'
		]
		gtf_generator = generate_genomic_regions(first_seven_genes)
		extracted_genes = [next(gtf_generator)[-1] for _ in range(7)]

		assert all(first_seven_genes == extracted_genes)

		# zcat data/Homo_sapiens.GRCh38.113.gtf.gz | grep -E '\Wgene\W' | wc -l
		n_gene_entries = 78932
		i = 7
		for _ in gtf_generator:
			i += 1
		assert n_gene_entries == i


def test_embed_dna():
	fasta_path='data/GRCh38.d1.vd1.fa'
	test_regions = [
		('1', 9121, 26894),
		('1', 584945, 829989)
	]
	embed_size = (2000, 8000)
	dna_embedder = embed_dna(test_regions, fasta_path, embed_size)

	for c, s, e in test_regions:
		chrom, embd_start, embd_end, dna_embedding = next(dna_embedder)
		# assert c == chrom and embd_start == s - embed_size[0] and embd_end == e + embed_size[1]
		assert dna_embedding.shape == (4, sum(embed_size))


def test_embed_atac():
	atac_path='data/overlap_genes_peaks.tsv'
	test_regions = [
		('1', 9121, 26894),
		('1', 584945, 829989)
	]
	embed_size = (2000, 8000)

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

	atac_embedder = embed_atac(atac_df, test_regions, embed_size)

	for c, s, e in test_regions:
		chrom, embd_start, embd_end, atac_embedding = next(atac_embedder)
		# assert c == chrom and embd_start == s - embed_size[0] and embd_end == e + embed_size[1]
		assert atac_embedding.shape == (1, sum(embed_size))


def test_embed_cnv():
	epianeu_path='data/epiAneuFinder_results.tsv'
	test_regions = [
		('1', 9121, 26894),
		('1', 584945, 829989)
	]
	embed_size = (2000, 8000)
	cnv_embedder = embed_cnv(epianeu_path, test_regions)
	for _ in test_regions:
		barcode, cnv_embedding = next(cnv_embedder)
		assert cnv_embedding.shape == (2, sum(embed_size)), "{} wrong CNV shape".format(barcode)


# def test_embed():
# 	gft_path='data/Homo_sapiens.GRCh38.113.gtf.gz'
# 	fasta_path='data/GRCh38.d1.vd1.fa'
# 	overlap_path='data/overlap_genes_peaks.tsv'
# 	epianeu_path='data/epiAneuFinder_results.tsv'
# 	embedder = embed(fasta_path, overlap_path, epianeu_path,gtf_path=gft_path)
# 	x = next(embedder)
# 	print(x)
# 	print(x.shape)
