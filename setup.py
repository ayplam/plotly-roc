# Copyright (c) 2020 ayplam

from plotly_roc import __version__

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="plotly-roc",
    version=__version__,
    author="Adrian Lam",
    author_email="ayplam@gmail.com",
    packages=["plotly_roc"],
    url="https://github.com/ayplam/plotly-roc",
    license="GPLv2",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python :: 3.6",
    ],
    install_requires=["pandas>=1.0.0", "plotly>=4.5.4,<4.6"],
    description="interactive roc curves with plotly",
    zip_safe=True,
)
