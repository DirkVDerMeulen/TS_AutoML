#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import setup, find_packages

NAME = 'TS_AutoML'
DESCRIPTION = 'Automated Machine Learning implementation for Time Series Prediction problems.'
URL = 'https://github.com/DirkVDerMeulen/TS_AutoML'
EMAIL = 'dirk-vandermeulen@hotmail.com'
AUTHOR = 'Dirk van der Meulen'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None

# Necessary requirements for package
REQUIRED = [
    'numpy',
    'pandas',
    'sklearn'
]

# Load the package's __version__.py module as a dictionary.
here = os.path.abspath(os.path.dirname(__file__))
about = {}

if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# Where the magic happens
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    license='unlicense',
    packages=find_packages(),
    install_requires=REQUIRED,
    include_package_data=True,
    zip_safe=False
)
