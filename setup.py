#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

NAME = 'TS_AutoML'
DESCRIPTION = 'Automated Machine Learning implementation for Time Series Prediction problems.'
URL = 'https://github.com/DirkVDerMeulen/TS_AutoML'
EMAIL = 'dirk-vandermeulen@hotmail.com'
AUTHOR = 'Dirk van der Meulen'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.1'

# Where the magic happens
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    license='unlicense',
    packages=['TS_AutoML'],
    zip_safe=False
)
