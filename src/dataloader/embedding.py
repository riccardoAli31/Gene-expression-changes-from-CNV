#!/bin/python

"""
Script for creating embeddings from DNA regions.
See https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02811-x/figures/5 
"""

import numpy as np
from numpy import ndarray
from typing import List, Tuple


def embed(dna_string: str, exon_start_stop: List[Tuple[int, int]],
          alphabet="ACGT") -> ndarray:
  """
  Encode a DNA string using one-hot encoding with two additional
  channels indicating the fist base of each reading frame and
  the first base after each 5' splice site. 
  """

  codon_size = 3
  # TODO respect introns? if yes we need a list of start/stop positions for reading frame 

  one_hot_dna = np.zeros((len(alphabet), len(dna_string)), dtype='u1')
  for i, aa in enumerate(dna_string):
    one_hot_dna[alphabet.index(aa), i] = 1

  exon_structure = np.zeros(len(dna_string), dtype='u1')
  for start, stop in exon_start_stop:
    exon_structure[start:stop] = 1

  # compute reading frame from exon structure
  reading_frame = np.zeros(len(dna_string), dtype='u1')
  overhead = 0
  for start, stop in exon_start_stop:
    reading_frame[list(start + (codon_size - overhead), stop, codon_size)] = 1
    overhead = (stop - start + overhead) % 3

  return np.vstack([one_hot_dna, reading_frame, exon_structure])

