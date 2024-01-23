# Contribution

## Development

Install this project, the required dependencies and the pre-commit hooks for development.

```note
You have to clone this project first and run the following commands in the root folder of this project.
```

```bash
pip install -e ".[dev]"
pre-commit install
```

### Build Documents

Install the required dependencies to build the documents.

```bash
pip install -e ".[doc]"
```

Launch the live server to build and preview the documents.

```bash
sphinx-autobuild docs docs/_build
```

### Build Package

Install the required dependencies to build the package.

```bash
pip install -e ".[build]"
```

Build the package.

```bash
python -m build
```

### Unit Tests

Install the required dependencies to run unit tests.

```bash
pip install -e ".[test]"
```

#### Run the unit tests.

```bash
pytest
```

#### Run the unit tests with coverage.

```bash
pytest --cov=.
```

#### Run the unit tests with various python version.

```bash
tox -- --cov=.
```

## Code Style and Git Hooks

The code is formatted and linted by [ruff](https://github.com/astral-sh/ruff). The docstring is in [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and formatted by [docformatter](https://github.com/PyCQA/docformatter). The spelling is checked by [codespell](https://github.com/codespell-project/codespell). All commit messages should follow the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style to generate the changelog and semantic version automatically.

All the above requirements are checked by [pre-commit](https://pre-commit.com/) hooks. You can install them by `pre-commit install` after you clone the repo.

## Version, Tag and Release

We use [commitizen](https://github.com/commitizen-tools/commitizen) to bump the semantic version, create tags and generate the changelog automatically according to the commit messages. Then, [setuptools_scm](https://github.com/pypa/setuptools_scm) is used to generate the version number from the tags. Therefore, all commit messages should follow the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style to facilitate the version management tools.

Fortunately, the [commitizen](https://github.com/commitizen-tools/commitizen) can be used to generate the commit messages in the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style. The whole process of contribution is as follows:

- Install commitizen. You can install it by `pip install commitizen`, or if you have installed the `[dev]` extra dependencies, you can skip this step since [commitizen](https://github.com/commitizen-tools/commitizen) has already been included in the `[dev]` extra dependencies. If you are using [vscode](https://code.visualstudio.com/), you can install the [Commit Message Editor](https://marketplace.visualstudio.com/items?itemName=adam-bender.commit-message-editor) extension to generate the commit message following the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style.
- Make changes.
- Commit changes. Then you can use `cz commit` to commit your changes. It will guide you to write the commit message in the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) style.
- Bump version (Optional). If you have made some commits related to codes with `feat`, `fix`, `refactor`, `perf` or `BREAKING CHANGE` category, you can use `cz bump` to bump the version and generate the changelog automatically.
- Push changes. Then you can push your changes to the remote repo. If you have bumped the version, you should push the tags to the remote repo by `git push --tags`. If you are using the [vscode](https://code.visualstudio.com/), you can set `git.followTagsWhenSync` to `true` to automatically push the tags when you run the sync command.
