# scPPR
## Introduction
scPPR is an eigengene based bayesian Markov-Chain Monte Carlo approach for cell cycle pseudotime estimation from single-cell RNA-seq data. scPPR uses a set of eigengenes which conform to different cosine function curves within one cycle of the oscillation (0-2 π) and were extracted form a scRNA-seq expression matrix. After obtaining a group of eigengenes mentioned above, statistical methods are used to determine their phases, and then precise position in a base circle of cells were calculated. Based on continuous cell cycle pseudotime, the division of three discrete cycle stages can be performed. scPPR can be used to perform downstream analysis to discover new functions and assess its performance.

![Illustration](docs/illustration.PNG)

## Requirements
- numpy
- scipy
- pandas
- sklearn
- matplotlib

## Installation
Download scPPR by
````
git clone https://github.com/luhf611/scPPR.git
````
Intallation has been tested in a Linux platform with Python3.7

## Tutorials

We provide several [notebooks](https://github.com/luhf611/scPPR/tree/main/notebooks) with real datasets, which shows to start with an expression matrix, then preprocess data, and finally calculate the circular pseudotime.

## License
This project is licensed under the MIT License.