library(data.table)
# library(Seurat)
library(Matrix)
library(copykat)

data_root <- file.path("data/")
dir(data_root)

metadata <- fread(file.path(data_root, "GSE240822_GBM_ccRCC_RNA_metadata_CPTAC_samples.tsv"))
sample_names <- c("C3N-00495-T1_CPT0078510004_snRNA_ccRCC", "C3L-00004-T1_CPT0001540013_snRNA_ccRCC")
sample_name <- sample_names[1]

str(metadata[`GEO.sample` == sample_name])

data_path <- file.path(data_root, sample_name, "outs", "raw_feature_bc_matrix")
counts <- readMM(file.path(data_path, "matrix.mtx.gz"))
barcodes <- fread(file.path(data_path, "barcodes.tsv.gz"), header=FALSE)
colnames(counts) <- barcodes$V1
regions <- fread(file.path(data_path,"features.tsv.gz"),header=FALSE)
rownames(counts) <- paste(regions$V2)

filtered_counts <- counts[, colnames(counts) %in% metadata[`GEO.sample` == sample_name, Barcode]]
raw_mat <- as.matrix(filtered_counts)

saveRDS(raw_mat, file = file.path("data", "tmp", paste0(sample_name, "-raw_mat", ".RDS")))

copykat_out <- copykat(rawmat=raw_mat, sam.name=sample_name, id.type="S", genome = "hg20")

str(copykat_out)

saveRDS(copykat_out, file = file.path("data", "tmp", paste0(sample_name, "-copykat_out", ".RDS")))
