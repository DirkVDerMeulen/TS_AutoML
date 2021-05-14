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

# What packages are required for this module to be executed?
# Use the Dataiku packaged library versions here
# Other libraries can be used freely
REQUIRED = [
    "numpy==1.19.5",
    "pandas==1.1.5",
    "sklearn==0.0"
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['TS_AutoML'],
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['TS_AutoML'],

    install_requires=REQUIRED,
    include_package_data=True
)
