# Script to install R software for CNV extraction with
#  CopyKat and epiAneufinder for Unix based OS.
# Note: CopyKat depends on the igraph package, which
#  does not work well with conda environments. Thus,
#  please install this to your machine or a container.
sudo apt install r-base r-essentials r-cran-irkernel

# install R packages from separate script file
Rscript install.R
