# MHCpG
Predicting DNA methylation with sequence based deep-learning model with MeDIP-seq and histone modification
# Introduction
MHCpG is a sequence-based deep leraning ramework to predict DNA methylation sites with MeDIP-seq and histone modification information. Traditional sequence based deep learning model utilized only sequence information but failed to consider other factors that may reflect or impact DNA methylation. In this work, we proposed a sequence-based model with both MeDIP-seq data and Histone infor-mation to predict DNA methylated CpG sites (MHCpG). It turned out that using either MeDIP-seq data or histone modification data with sequence information could improve the performance, while combining three data together gave the best results.In addition, we used a collateral convolution layer with multiple convolutional kernels rather than multiple layers to extract features from input data.

# Prerequisites
- Python (2.7). [Python 2.7.13](https://www.python.org/downloads/release/python-2713/) is recommended.
- Numpy
- Keras
- Scipy

# Data
- All the sequence data used for training and testing can be downloaded from data file folder. It contains all the bed files of DNA methylation around all the chromosomes.
- The histone modification data we used can be downloaded from [here](http://hgdownload.cse.ucsc.edu/goldenpath/hg19/encodeDCC/wgEncodeBroadHistone/)
- The MeDIP-seq data we used can be retrieved from [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE56774)

# Usage
- To implement the model, run bash file run_both_histone_medip.sh
