## classification-utils

Helper functions for text classification with BERT.

## Installation

This code requires Python 3.7 venv and pip.

To install, run `./run install`.

**Optional**: Install GPU support with `./run pip install tensorflow-gpu==1.15.0`.

### Data preparation

Read data and standardize the format with `./run python prepare.py DATA_ID`

Multiple labels are split, sorted and represented with one-hot encoding.
The processed data is stored as _.pkl_ file.

## Coding support (mypy, format, lint)

The code is yapf / mypy / flake8 linted.

To run the linter, `./run lint`.

To format all files, `./run format`.

### MyPy

This package uses MyPy for Python type checking and intellisense.

To install mypy in vscode, install the 'mypy' plugin and run these:

```sh
sudo apt install python3.8-venv python3.8-dev
python3.8 -m venv ~/.mypyls
~/.mypyls/bin/pip install -U wheel
~/.mypyls/bin/pip install -U "https://github.com/matangover/mypyls/archive/master.zip#egg=mypyls[patched-mypy]"
```
