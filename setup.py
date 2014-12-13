#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Dec 11, 2014.

from setuptools import setup

setup(
    name = "xy_deep_learning",
    version = "0.1.dev",
    url = "https://github.com/yxiong/xy_deep_learning",
    author = "Ying Xiong",
    author_email = "yxiong@seas.harvard.edu",
    description = "Utilities for deep learning.",
    packages = ["xy_deep_learning",],
    install_requires = ["numpy", "theano", "xy_python_utils"]
    )
