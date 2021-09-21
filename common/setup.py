#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

VERSION = imp.load_source("", "wm_utility/version.py").__version__

setup(
    name="wm_utility",
    author=["Michael W. Reimann"],
    author_email="michael.reimann@epfl.ch",
    version=VERSION,
    description="Dealing with wm-style connectomes",
    long_description="Dealing with wm-style connectomes",
    url="http://bluebrain.epfl.ch",
    license="LGPL-3.0",
    install_requires=["numpy",
                      "pandas",
                      "voxcell"
                      ],
    packages=find_packages(),
    scripts=[
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
