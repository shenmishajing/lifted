# Project Template

This template project is based on [pytorch lightning](https://pytorch-lightning.readthedocs.io/en/stable/) and [lightning-template](https://github.com/shenmishajing/lightning_template). Please read the docs of them before using this template.

- Pre-written docs for installation, contribution etc, with [readthedocs](https://readthedocs.org/) support. See the `docs` folder and the `.readthedocs.yaml` file for more details. To use this feature, you need to import your GitHub repo on readthedocs. See their [docs](https://docs.readthedocs.io/en/stable/index.html) for more details.
- Pre-commit hooks for formatting, linting, style checking and commit messages checking. See the `.pre-commit-config.yaml` file for more details.
- Unit tests with [pytest](https://docs.pytest.org/en/latest/), [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) and [tox](https://tox.wiki/en/latest/) support. Remember to add your tests to the `tests` folder before you want to run them. See the `pyproject.toml` file and `tox.ini` file for more details.
- Auto-generated semantic version tags and the changelog file based on commit messages following the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style. However, this feature requires all the commit messages to follow the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style. See the [contribution docs](docs/get_started/contribution.md) for more details.
- Auto-build and release with [setuptools_scm](https://github.com/pypa/setuptools_scm) support leveraging the auto-generated semantic version tags, which is useful when you want to release your project as a Python package. Remember to change the package name before you want to release it.
- Github action support for all the above features including checking pre-commit hooks, unit tests and auto-releasing. Only the pre-commit hooks checker is enabled by default, you can enable the other github actions by moving them from the `.github/examples` folder to the `.github/workflows` folder.

Please read the [usage docs](docs/get_started/usage.md) for more details on how to use this template project.

````{grid} 2
:gutter: 3

  ```{grid-item-card} Installation Guides

  Get started by installing Lightning Template.

  +++

    ```{button-ref} get_started/installation
    :expand:
    :color: secondary
    :click-parent:

    To the installation guides
    ```
  ```

  ```{grid-item-card} Usage Guides

  Details of the core features provide by Lightning Template.

  +++

    ```{button-ref} get_started/usage
    :expand:
    :color: secondary
    :click-parent:

    To the Usage guides
    ```
  ```

  ```{grid-item-card} API Reference Guides

  References of the API provided by Lightning Template.

  +++

    ```{button-ref} autoapi/project/index
    :expand:
    :color: secondary
    :click-parent:

    To the API Reference guides
    ```
  ```

  ```{grid-item-card} Contribution Guides

  Helpful instruction to develop Lightning Template.

  +++

    ```{button-ref} get_started/contribution
    :expand:
    :color: secondary
    :click-parent:

    To the Contribution guides
    ```
  ```
````

```{toctree}
:hidden:
:caption: Get Started

get_started/installation
get_started/usage
get_started/contribution
get_started/changelog
```

```{toctree}
:hidden:
:caption: API References

autoapi/project/index
```
