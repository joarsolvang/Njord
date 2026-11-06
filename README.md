# Project Title
_Put your project description here_

[![CI](https://github.com/res-ds/ds-rnd-python-template/actions/workflows/ci.yaml/badge.svg)](https://github.com/res-ds/ds-rnd-python-template/actions/workflows/ci.yaml)

## Usage
Provide a description of the project here. If the project is intended to be distributed as a package, include instructions on how the package should be installed and used.

To use this template, simply rename `example-pkg` and `example_pkg` to the name of your choice, including everywhere in `pyproject.toml`.

## Development
The development environment should be created and managed using [uv](https://docs.astral.sh/uv/). To create the environment:
```commandline
uv sync
```
To run the formatting, linting and testing:
```commandline
uv run poe all
```
Or simply
```commandline
poe all
```
if you have activated the virtual environment (VSCode will do this automatically for you). For example, to activate the environment from a PowerShell prompt:
```powershell
. ".venv\Scripts\activate.ps1"
```

## Contact
Put your contact details here.
