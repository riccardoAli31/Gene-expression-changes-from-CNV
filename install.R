# R package installation
# Note: this script asserts you already installed r-base and
#  r-essentials from your respective OS sotware repositrory.
install.packages(c("devtools", "BiocManager", "ggdendro", "IRkernel"), repos = "https://ftp.fau.de/cran/")
BiocManager::install(c("GenomicAlignments", "SummarizedExperiment", "plyranges", "Rsamtools", "GenomeInfoDb", "BSgenome.Hsapiens.UCSC.hg38", "GenomicRanges", "Biostrings", "BiocGenerics", "S4Vectors", "GenomicFeatures"))
library(devtools)
install_github("colomemaria/epiAneufinder")
install_github("navinlabcode/copykat")
