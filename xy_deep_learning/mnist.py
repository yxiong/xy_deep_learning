#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Dec 09, 2014.

import numpy as np
import theano
import theano.tensor as T
import time

from xy_python_utils import os_utils

from classifiers import *
from utils import *

def logistic_sgd(learning_rate = 0.13,
                 num_epochs = 1000,
                 batch_size = 600):
    """Logistic regression with stochastic gradient descent.

    Should yield the same results as
    'DeepLearningTutorial/code/logistic_sgd.py'.

    """
    # Load data.
    print "Loading data..."
    datasets = load_mnist_data()

    # Build model.
    print "Building model..."
    classifier = LogisticRegression(dim_in = 28*28, dim_out = 10)
    models, num_batches = create_minibatch_models(
        classifier, datasets, batch_size, learning_rate)

    # Train model.
    print "Training the model..."
    train_model(models, num_batches, num_epochs, patience=5000)

    os_utils.mkdir_p("trained-models")
    classifier.save_params("trained-models/logistic_sgd.dat")

def mlp(learning_rate = 0.01, L1_reg = 0.00, L2_reg = 0.0001, num_epochs = 1000,
        batch_size = 20, num_hidden = 500):
    """Multi-layer perceptron.

    Should yield the same results as 'DeepLearningTutorial/code/mlp.py'.

    """
    # Load data.
    print "Loading data..."
    datasets = load_mnist_data()

    # Build model.
    print "Building model..."
    x = T.matrix('x')
    y = T.ivector('y')
    rng = np.random.RandomState(1234)
    classifier = MultiLayerPerceptron(
        dim_in=28*28, dim_out=10, dim_hidden=num_hidden, rng=rng)
    cost = classifier.negative_log_likelihood(x, y) + \
           L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
    models, num_batches = create_minibatch_models(
        classifier, datasets, batch_size, learning_rate,
        x = x, y = y, cost = cost)

    # Train model.
    print "Training the model..."
    train_model(models, num_batches, num_epochs, patience=10000)

    os_utils.mkdir_p("trained-models")
    classifier.save_params("trained-models/mlp.dat")

def convolutional_mlp(learning_rate=0.1, num_epochs=200,
                    num_kernels=[20, 50], batch_size=500):
    """LeNet5 convolutional neural network.

    Should yield the same results as
    'DeepLearningTutorial/code/convolutional_mlp.py'.

    """
    rng = np.random.RandomState(23455)

    # Load data.
    print "Loading data..."
    datasets = load_mnist_data()

    # Build model.
    print "Building model..."
    dim_in = (batch_size, 1, 28, 28)
    classifier = LeNet5(
        dim_in = dim_in,
        dim_out = 10,
        num_kernels = num_kernels,
        rng = rng)
    models, num_batches = create_minibatch_models(
        classifier, datasets, batch_size, learning_rate)

    # Train model.
    print 'Training the model...'
    train_model(models, num_batches, num_epochs, patience=10000)

    os_utils.mkdir_p("trained-models")
    classifier.save_params("trained-models/convolutional_mlp.dat")
