#!/usr/bin/env python
import os.path
import sys

from setuptools import find_packages, setup

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

sys.path.insert(0, this_directory)

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='huridocs-classification-utils',
    version=0.1,
    description='Classifications utils',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/huridocs/classification-utils',
    author='Huridocs',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={},
    python_requires='>=3.6',
    entry_points={},
)
