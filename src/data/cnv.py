from typing import List, Tuple
from ..util import interval_overlap
from pandas import DataFrame

def extract_cnv_overlaps(cnv_df: DataFrame, barcode: str, 
						 region: Tuple[str,int,int]
						 ) -> List[Tuple[Tuple[int, int],int]]:
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



@DeprecationWarning
def gernerate_cnv_status_per_barcode(file_path: str):
	"""
	Generator for creating CNV status per barcode.
	This file is in tsv format with the columns:
	idx seq start end cell-<BARCODE_ID> ...

	file_path : str of path to file with CNV classifications
	"""
	
	barcode = None
	with open(file_path, 'r') as in_file:
		for line in in_file:
			pass

