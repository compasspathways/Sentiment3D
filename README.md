# COMPASS Pathways Three-dimensional Sentiment Model

<div align="center">

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/compasspathways/SentimentDD/blob/main/.pre-commit-config.yaml)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

A package for computing the three-dimensional sentiment scores and a Jupyter notebook for replicating the figures in the paper "[From a Large Language Model to Three-Dimensional Sentiment](https://LINK.TO.PREPRINT)".

</div>

## Contents
The sentiment model code is in "sentiment3d" and an example that demonstrates the use of this model is in the notebook example.ipynb. To reproduce all the figures in our paper, have a look at sentiment3d_paper.ipynb.

## Installation

```bash

python -m pip install git+https://github.com/compasspathways/Sentiment3D
```

## Contributing

Install from source and run in a virtual env:

```bash
git clone git@github.com:compasspathways/Sentiment3D.git
cd Sentiment3D
python3 -m venv ./s3d_env
source ./s3d_env/bin/activate
python -m pip install -e .
```

To run the notebooks:

```bash
python -m pip install jupyterlab ipywidgets
python -m ipykernel install --name "s3d_env" --user

```

After you start the notebook server, be sure to switch to the s3d_env kernel.

Before making any pull requests, be sure to install dev requirements and add the pre-commit hooks:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Citation

Please cite our paper titled "[From a Large Language Model to Three-Dimensional Sentiment](https://LINK.TO.PREPRINT)".
