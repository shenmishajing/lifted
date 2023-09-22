## Installation

### Manually installation

First, install some packages from their official site manually, mainly some packages related to cuda, and you have to choose the cuda version to use. 

#### Pytorch

Install [pytorch](https://pytorch.org/get-started/locally/) from their official site manually. You can skip this if you want to use the latest pytorch.

### Automaticaly installation

Generally, you can just use the latest pacages in `requirements.txt` without specific their version, so you can use command as follow to install this project and all required packages.

```bash
pip install -r requirements.txt
pip install -e .
```

### Install by conda

If you want to use conda, you can use the following command to create a new environment and install all required packages.

```bash
conda env create --file requirements/conda/conda.yaml
```

or

```bash
conda create --name <env_name> --file requirements/conda/conda.txt
```

then, install this project by

```bash
pip install -e .
```
