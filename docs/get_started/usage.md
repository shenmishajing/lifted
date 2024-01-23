# Usage

## Installation

See the [Installation with pip](installation.md#installation-with-pip) section in the [installation docs](docs/get_started/installation.md) for details.

### Generate the conda.yml

If you want to spread out your project, you may need a `conda.yml` to help users install your project and all required packages. You can use the following command to export your conda environment to the `conda.yml`.

````{note}
Make sure you are in the correct conda environment before you run the following command.

You can enter the correct conda environment now by:

  ```bash
  conda activate <env_name>
  ```
````

```bash
conda env export --no-builds | grep -v "prefix" > requirements/conda.yml
```

## Development

### Models and Datasets

Leveraging the powerful [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [lightning-template](https://lightning-template.readthedocs.io), we can focus on the design of our models and the implementation of our datasets only without any other boilerplate engineering code including the loggers, the config management, the training loop, etc. Refer to the [cli doc](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli) of [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) for more details on how to launch your models and datasets in your config files.

### Config files

To run an experiment, you have to write a config file. Please refer to [config files doc](https://lightning-template.readthedocs.io/en/latest/get_started/usage.html#config-files) of [lightning-template](https://lightning-template.readthedocs.io) for more details.

### Experiments

All our experiments run through a command line script called `cli`. You can refer to the [cli doc](https://lightning-template.readthedocs.io/en/latest/tools/cli.html) from [lightning-template](https://lightning-template.readthedocs.io) for more details.

### Debug the build of docs, the release and unit tests

If you want to debug the build of docs, the release and unit tests, you can refer to the [contribution doc](contribution.md) for more details on them.

### Commit messages

We recommend you follow the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style for commit messages. You can refer to [contribution doc](contribution.md) for more details.

### Pre-commit Hooks

Pre-commit hooks are used to automatically format, lint your commit and check your commit messages. You can refer to [contribution doc](contribution.md) for more details.

## Deployment

### Documentation

If you want to build a documentation site for your project, you have to follow the following steps:

- Import your project from [readthedocs](https://readthedocs.org/).
- In `Admin > Automation Rules` of the Setting of your project, add a new rule to automatically activate the new version when a new tag is published. The `Match` of the rule should be `SemVer versions`. The `Version type` should be `tag` and the `Action` should be `Activate version`.
- Make a new name for your project.
- Modify the `project` and `author` information in `docs/config.py`.
- Rewrite an introduction and rename the project in `docs/index.md`
- Rewrite the Usage in `docs/get_started/usage.md`
- Modify the name and URLs used in `README.md`.
- Add more docs as you want to the `docs/` folder and update the `toctree` in `docs/index.md`.
- Commit the changes and push them to the remote repo.
- Done. You have successfully deployed the documentation of your project, and you can see a new version when you push a new tag to the remote repo.

### Unit tests

If you want to add unit tests for your project, you have to follow the following steps:

- [Optional if your repo is public] Setup up your project on [codecov](https://app.codecov.io/) and set up the `CODECOV_TOKEN` secret for your project. You can skip this step if your project is public since the upload action of [codecov](https://app.codecov.io/) supports tokenless upload for public repo now. But if your project is private, it's required to set up the `CODECOV_TOKEN` secret for your project.
- Add unit tests in the `tests` folder following the [pytest](https://docs.pytest.org/en/latest/) style.
- Move the `.github/examples/test.yml` to the `.github/workflows` folder.
- [Optional] Check the Python versions and os var set in the `.github/workflows/test.yml` file. You can add more versions or remove them if you want.
- Commit your changes and push them to the remote repo.
- Enable flag analysis in the `Flags` setting of your project on [codecov](https://app.codecov.io/).
- Done. You can see the test results and coverage reports in the [codecov](https://app.codecov.io/) dashboard.

### Release

If you want to publish your project as a python package and also generate github release automatically, you have to follow the following steps:

- Tag your project with `1.0.0`.
- Make up a new name for your project. Modify the name in `pyproject.toml` and rename the `src/project` folder.
- Modify the `release` and `autoapi_dirs` information in `docs/config.py`, according to your new name.
- Modify the name and URLs used in `README.md` and `pyproject.toml`.
- Add any content you want in the `src` folder.
- Move the `.github/examples/release.yml` and `.github/examples/pre_release.yml` to the `.github/workflows` folder.
- Commit your changes following the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style.
- Bump version by [commitizen](https://github.com/commitizen-tools/commitizen) using the `cz bump` command. A changelog file will be generated in the `docs/get_started` folder automatically according to your commit messages in the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style.
- Add your project as a publisher on [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/) respectively. You should set up the `release.yml` for [PyPI](https://pypi.org/) and the `pre_release.yml` for [TestPyPI](https://test.pypi.org/).
- Push your changes and tags to the remote repo.
