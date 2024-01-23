# Installation

## Install project and dependencies

### Install with conda

We recommend you use `conda` to install this project and all required packages with the specific version to recurrent our experiments and results of them. The following commands can be used to install this project and all required packages.

```bash
conda env create -f requirements/conda.yml -n <env_name>
conda activate <env_name>
pip install -e .
```

```{warning}
It is worth noting that you have to make sure that the cuda version of your machine is consistent with the version of the packages in `conda.yml`, since we have set the Pytorch version and cuda version in the `conda.yml`.
```

### Installation with pip

```{warning}
Notably, creating a new conda environment is the most recommended way to install our project and all the required packages to recurrent our results, since you can keep the same version of every package with ours. Installing our project with `pip` may result in different results with different versions of the packages. However, if you counter some problem in the installation process of `conda` method or you have to use a different cuda version, you can try to use the `pip` method to install this project and all required packages.
```

When installing this project with `pip`, you can choose the version of some packages you want to use, mainly some packages related to cuda, like [Pytorch](https://pytorch.org/get-started/locally/). See the following sections for details.

#### Python

We recommend you use the latest version of Python, which works well generally and may provide a better performance. The minimum supported version of Python is `3.8`.

#### Pytorch

Install [Pytorch](https://pytorch.org/get-started/locally/) from their official site manually. You have to choose the version of Pytorch based on the cuda version on your machine. Similarly, we recommend you use the latest version of Pytorch, which works well generally and may provide a better performance. You can skip this step if it's fine to use the latest version of Pytorch and the `pip` will install it in the next section. The minimum supported version of Pytorch is `1.11`.

#### This project and its dependencies

Generally, you can just use the latest dependencies without a specific version, so you can use the command as follows to install this project and all required packages.

```bash
pip install -e ".[optional]"
```

## Logger

### Wandb

By default, we use `WandbNamedLogger` as the logger. To use `Wandb`, you have to create an account on their [site](https://wandb.ai/) and login following their [doc](https://docs.wandb.ai/quickstart).

### Other loggers

You also can use any other logger you want to use. For example, you can use the `TensorBoardLogger` instead of the `WandbNamedLogger`. See the `default_runtime.yaml` under the `configs` folder for more details.
