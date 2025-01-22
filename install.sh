# Script to install R software for CNV prediction project
# CNV extraction is done with CopyKat and epiAneufinder and
#  installed separately with a Rscript.
conda create -n cmscb r-base r-essentials python pytorch numpy pandas tqdm torchvision plotnine pytest
conda activate cmscb

# install R packages from separate script file
Rscript install.R
