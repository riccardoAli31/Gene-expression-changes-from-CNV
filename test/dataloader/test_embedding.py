import numpy as np
from src.dataloader.embedding import encode_one_hot_dna, \
    encode_open_chromatin, encode_CNV_regions, embed

def test_encode_one_hot_dna():
	# short test case
	dna_seq = "CGATGCTGTGATGC"
	expected_result = np.array([
		# one hot encoding
		[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
		[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
		[0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
		[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
	])
	assert np.array_equal(encode_one_hot_dna(dna=dna_seq), expected_result)

	# simple padding test case
	dna_seq = "NNNTGCTGTGANNN"
	expected_result = np.array([
		# one hot encoding
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
		[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
		[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
	])
	assert np.array_equal(encode_one_hot_dna(dna=dna_seq), expected_result)

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
	])
	assert np.array_equal(encode_one_hot_dna(dna=dna_seq), expected_result)

def test_encode_open_chromatin():
	# TODO: account for 0- to 1-based difference and end-inclusive intervals
	#  use correct=-1 for relative_idx
	upsteam = 2000
	downsteam = 8000

	def relative_idx(i, emb_start, emb_end, correct=0):
		if emb_end < i:
			i = emb_end
		assert i >= emb_start and i <= emb_end
		return i - emb_start + correct
	
	# simple one peak case
	embed_interval = [0, 20]
	peak_intervals = [(5, 10)]
	arr_expected = np.zeros(embed_interval[1] - embed_interval[0], dtype='u1')
	for start, end in peak_intervals:
		rel_start = relative_idx(start, *embed_interval)
		rel_end = relative_idx(end, *embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert np.array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)

	# simple two peak case
	embed_interval = [0, 20]
	peak_intervals = [(5, 10), (13, 17)]
	arr_expected = np.zeros(embed_interval[1] - embed_interval[0], dtype='u1')
	for start, end in peak_intervals:
		rel_start = relative_idx(start, *embed_interval)
		rel_end = relative_idx(end, *embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert np.array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)

	# simple four peak case
	embed_interval = [0, 20]
	peak_intervals = [(1, 3), (5, 10), (13, 17), (19, 22)]
	arr_expected = np.zeros(embed_interval[1] - embed_interval[0], dtype='u1')
	for start, end in peak_intervals:
		rel_start = relative_idx(start, *embed_interval)
		rel_end = relative_idx(end, *embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert np.array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)

	# single peak case: ENSG00000290825
	embed_interval = [9121 - upsteam, 9121 + downsteam]
	peak_intervals = [(9855, 10676)]
	arr_expected = np.zeros(embed_interval[1] - embed_interval[0], dtype='u1')
	for start, end in peak_intervals:
		rel_start = relative_idx(start, *embed_interval)
		rel_end = relative_idx(end, *embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert np.array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)

	# multiple peak case: ENSG00000228794
	embed_interval = [823138 - upsteam, 823138 + downsteam]
	peak_intervals = [
		(822918, 823607), (825354, 826159), (827067, 827928), (835242, 836034),
		(838089, 838689), (851462, 852319), (854606, 855500), (869498, 870402)
	]
	arr_expected = np.zeros(embed_interval[1] - embed_interval[0], dtype='u1')
	for start, end in peak_intervals:
		if start > embed_interval[1]:
			continue 
		rel_start = relative_idx(start, *embed_interval)
		rel_end = relative_idx(end, *embed_interval)
		arr_expected[rel_start:rel_end] = 1
	assert np.array_equal(encode_open_chromatin(embed_interval, peak_intervals), arr_expected)


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

# def test_encode_CNV_regions():
#       # TODO
#       pass


def test_embed():
	gft_path='data/Homo_sapiens.GRCh38.113.gtf.gz'
	fasta_path='data/GRCh38.d1.vd1.fa'
	overlap_path='data/overlap_genes_peaks.tsv'
	embedder = embed(gft_path, fasta_path, overlap_path)
	x = next(embedder)
	print(x)
