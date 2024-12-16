# script to install software for cmscb project 8
conda create -n cmscb scanpy python-igraph leidenalg r-base r-essentials

# custom R packages with BioConductor
conda activate cmscb

# install R packages from separate script file
Rscript install.R

