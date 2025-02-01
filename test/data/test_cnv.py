from pandas import DataFrame, concat
from src.data import extract_cnv_overlaps


def test_extract_cnv_overlaps():
	starts = [300_001, 500_001, 600_001, 700_001, 800_001]
	ends = [400_000, 600_000, 700_000, 800_000, 900_000]
	barcode_status = {
		'cell-GCGCAATGTTGCGGAT-3': [1, 1, 1, 1, 1],
		'cell-CTAGTGAGTCACCTAT-3': [1, 1, 0, 0, 0],
		'cell-AATCATGTCGATCAGT-1': [1, 1, 1, 1, 1]
	}
	df = concat([
		DataFrame({
			'seq': ['1'] * len(starts),
			'start': starts,
			'end': ends,
		}), 
		DataFrame(barcode_status)],
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
