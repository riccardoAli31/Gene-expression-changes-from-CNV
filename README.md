# Predicting Gene Expression Changes from Copy Number Variation
This project aims to explore the contribution of copy number variations (CNVs) on gene expression changes.
For this we employ different machine learning models to predict a high or low gene expression on the basis of DNA sequence, opne chromatin and CNVs.

## Reproducitbility
### Data
To reproduce the analysis, please download the multi OMICs dataset from [10X Genomics](https://www.10xgenomics.com/welcome?closeUrl=%2Fdatasets&lastTouchOfferName=Human%20Kidney%20Cancer%20Nuclei%20isolated%20with%20Chromium%20Nuclei%20Isolation%20Kit%2C%20SaltyEZ%20Protocol%2C%20and%2010x%20Complex%20Tissue%20DP%20(CT%20Sorted%20and%20CT%20Unsorted)&lastTouchOfferType=Dataset&product=chromium&redirectUrl=%2Fdatasets%2Fhuman-kidney-cancer-nuclei-isolated-with-chromium-nuclei-isolation-kit-saltyez-protocol-and-10x-complex-tissue-dp-ct-sorted-and-ct-unsorted-1-standard).
Furthermore, please get the human reference genome version 38 \href{https://api.gdc.cancer.gov/data/254f697d-310d-4d7d-a27b-27fbf767a834}{fasta} and \href{https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/Homo_sapiens.GRCh38.113.gtf.gz}{gtf} file.
You may index the fasta file if you indend to recompute the embeddings.

### Software
Please use the conda environment `env.yml` from this git repo to rerun the code.

