from typing import Tuple


def relative_idx(idx: int, interval: Tuple[int,int],
				 correct: int = 0, clip_start=True,
				 clip_end=True) -> int:
	"""
	Calculates relative index inside an interval from an index based outside the interval.
	E.g. index=42, interval=(13, 50) -> relative index is 29
	This function is intended to be used on chromosomal indices and intervals.

	idx : int of index based outside the interval
	interval : Tuple[int,int] of interval start and end coordinates
	correct : int of index shift correction. I.e. from 0-based to 1-based use +1.
	clip_* : bool wether idx should be interval start/end, if it exceeds interval start/end.
		Otherwise an AssertionError will be raised.
	"""

	start, end = interval
	assert start < end

	if clip_start and idx < start:
		idx = start
	
	if clip_end and end < idx:
		idx = end
	
	assert start <= idx <= end
	
	return idx - start + correct


def interval_overlap(interval_1: Tuple[int,int], 
		interval_2: Tuple[int,int]) -> Tuple[int,int]:
	"""
	Calculates the start and end position of an overlap 
	between two intervals.

	"""
	start_1, end_1 = interval_1
	start_2, end_2 = interval_2
	
	assert start_1 < end_1 and start_2 < end_2

	if end_1 < start_2 or end_2 < start_1:
		return None
	
	start = start_1 if start_2 <= start_1 else start_2
	end = end_1 if end_1 <= end_2 else end_2

	return (start, end)

