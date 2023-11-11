## Installation with pip (for developers)

Notably, creating a new conda environment is the most recommended way to install our project and all the required packages to recurrent our results, since you can keep the same version of every package with ours, and you will not get different results from ours due to the differences in the version of the python packages. This method is designed for the developers of this project instead of the users. If you want to develop this project or write a new project based on this, you can use this method to install this project and all required packages.

### Manual installation

First of all, you have to choose the version of some packages and install them from their official site manually, mainly some packages related to cuda, and also, you have to choose the cuda version. 

#### Python

We recommend you use the latest version of Python, which works well generally and may provide a better performance.

#### Pytorch

Install [Pytorch](https://pytorch.org/get-started/locally/) from their official site manually. You have to choose the version of Pytorch based on the cuda version on your machine. Similarly, we recommend you use the latest version of Pytorch, which works well generally and may provide a better performance. You can skip installing the Pytorch manually in this section and the `pip` will install it with the latest version in the next section if you want to use the latest Pytorch.

### Automatical installation

Generally, you can just use the latest packages in `requirements.txt` without specific their version, so you can use the command as follows to install this project and all required packages.

```bash
pip install -r requirements/pip.txt
pip install -e .
```

### Generate the conda.yml

If you want to spread out your project, you may need a `conda.yml` to help users install your project and all required packages. You can use the following command to export your conda environment to the `conda.yml`.

```bash
# make sure you are in the correct conda environment
# or enter the correct conda environment now by:
# conda activate <env_name>
conda env export --no-builds | grep -v "prefix" > requirements/conda.yml
```
