xy_deep_learning
================

Utilities for deep learning in Python.

The utilities are largely based on the deep learning tutorial by LISA lab in
University of Montreal ([website](http://deeplearning.net/tutorial/), [source
code](https://github.com/lisa-lab/deeplearningtutorials)). We basically refactor
the code into modules for better reusability and extensibility, and remove
lengthy documentations for better readibility.

Last access of
[lisa-lab/DeepLearningTutorials](https://github.com/lisa-lab/deeplearningtutorials)
on Dec 10, 2014, at its git commit `d4a96b52109bfeb1d3408d1ff08e90b7b92a084f`.

Author: Ying Xiong.  
Created: Dec 10, 2014.

Prerequisites
-------------

[`Theano`](http://deeplearning.net/software/theano/)

[`xy_python_utils`](https://github.com/yxiong/xy_python_utils)

Usage
-----

Install the package in development mode:

    python setup.py develop

Run tests on MNIST handwritten digits dataset:

    >>> import mnist
    >>> mnist.logistic_sgd()
    >>> mnist.mlp()
    >>> mnist.convolutional_mlp()


Visualize classification results on MNIST dataset.

    >>> from mnist import *
    >>> tr_set, vl_set, te_set = load_mnist_data()
    >>> visualize_data_xy(tr_set, 20)
    >>>
    >>> classifier = MultiLayerPerceptron.load_from_file(
            "trained-models/mlp.dat")
    >>> visualize_mistakes(classifier, te_set, 20)
