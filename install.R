# R package installation
install.packages(c("devtools", "BiocManager", "ggdendro"))
BiocManager::install(c("GenomicAlignments", "SummarizedExperiment", "plyranges", "Rsamtools", "GenomeInfoDb", "BSgenome.Hsapiens.UCSC.hg38", "GenomicRanges", "Biostrings", "BiocGenerics", "S4Vectors", "GenomicFeatures"))
library(devtools)
install_github("colomemaria/epiAneufinder")
install_github("navinlabcode/copykat")

