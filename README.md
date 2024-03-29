# COMPASS Pathways Three-dimensional Sentiment Model

<div align="center">

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/compasspathways/SentimentDD/blob/main/.pre-commit-config.yaml)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

A package for computing the three-dimensional sentiment scores and a Jupyter notebook for replicating the figures in the paper "[From a Large Language Model to Three-Dimensional Sentiment](https://psyarxiv.com/kaeqy)".

</div>

## Contents
The sentiment model code is in "sentiment3d" and an example that demonstrates the use of this model is in the notebook example.ipynb. To reproduce all the figures in our paper, have a look at sentiment3d_paper.ipynb.
The 3d sentiment model should run on any machine with 4GB of RAM or more.

## Installation

```bash

python -m pip install git+https://github.com/compasspathways/Sentiment3D
```

## Contributing

Install from source and run in a virtual env:

```bash
git clone git@github.com:compasspathways/Sentiment3D.git
cd Sentiment3D
python3 -m venv venv-s3d
source venv-s3d/bin/activate
pip install -e ."[explore,dev]"
pre-commit install
```

To run the notebooks:

```bash
python -m ipykernel install --name "venv-s3d" --user
```

After you start the notebook server, be sure to switch to the venv-s3d kernel.


## Interactive Demo

Please check out the interactive demo on Hugging Face at https://huggingface.co/spaces/compasspathways/Sentiment3D

## Citation

Please cite our paper titled "[From a Large Language Model to Three-Dimensional Sentiment](https://psyarxiv.com/kaeqy)".
