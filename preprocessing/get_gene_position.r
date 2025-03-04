library(rtracklayer)

# Specify the path to your GTF file
gtf_file <- '/Users/gajaj/OneDrive/Documents/TUM/computational_single_cell/Gene-expression-changes-from-CNV/preprocessing/Homo_sapiens.GRCh38.113.gtf'

# Import the GTF file
gtf_data <- import(gtf_file, format = "gtf")

gtf_data_genes <- gtf_data[gtf_data$type == "gene", ]

# Convert GRanges object to a data frame
gtf_data_df <- as.data.frame(gtf_data_genes)

# Add separate start and end columns from the ranges column
gtf_data_df$start <- start(gtf_data_genes)
gtf_data_df$end <- end(gtf_data_genes)

# Dataset: [cromosome, start, end, gene_id]
genes_positions <- gtf_data_df[, c("seqnames", "start", "end", "gene_id")]

write.csv(genes_positions, "/Users/gajaj/OneDrive/Documents/TUM/computational_single_cell/Gene-expression-changes-from-CNV/preprocessing/Multiome/gene_positions.csv", row.names = FALSE)





