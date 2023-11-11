## Introduction

## Installation

We recommend you use `conda` to install this project and all required packages with the specific version to recurrent our experiments and results of them. The following commands can be used to install this project and all required packages.

```bash
conda env create -f requirements/conda.yml -n <env_name>
conda activate <env_name>
pip install -e .
```

It is worth noting that you have to make sure that the cuda version of your machine is consistent with the version of the packages in `conda.yml`, since we have set the Pytorch version and cuda version in the `conda.yml`.

If you want to write a new project based on this project, you can refer to [installation with pip docs](docs/installation/installation_with_pip.md) for details. Also, if you run into any problems with the installation through conda, you can try the installation with pip, but there is no guarantee that you can recurrent our results if so.

## Prepare

See [prepare docs](docs/prepare/prepare.md) for details.

## Usage

All our experiments run through a command line script called `cli`. You can refer to the [cli doc](https://github.com/shenmishajing/lightning_template/blob/main/docs/tools/cli.md) from [lightning-template](https://github.com/shenmishajing/lightning_template) for more details.

## Experiment

See [experiment docs](docs/experiments/experiments.md) for details.
