import pytest
from src.util import relative_idx, interval_overlap


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
