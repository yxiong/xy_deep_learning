#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Dec 09, 2014.

import abc
import cPickle
import math
import numpy as np
import theano
import theano.tensor as T

class Classifier:
    """Base class for all classifiers."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, dim_in, dim_out, *args, **kw):
        """The initialize function should provide the field `params`."""

    @abc.abstractmethod
    def p_y_given_x(self, x):
        """Probability of y given x and internal states (`params`)."""
        return T.nnet.softmax(T.dot(x, self.W) + self.b)

    def classify(self, x):
        """Given `x`, predict the label `y`."""
        return T.argmax(self.p_y_given_x(x), axis=1)

    def negative_log_likelihood(self, x, y):
        """Compute negative log likelihood."""
        return -T.mean(T.log(self.p_y_given_x(x))[T.arange(y.shape[0]), y])

    def errors(self, x, y):
        """Compute 0-1 errors."""
        return T.mean(T.neq(self.classify(x), y))

    def save_params(self, filename):
        """Save parameters to a file."""
        with open(filename, 'wb') as f:
            for param in self.params:
                cPickle.dump(param.get_value(borrow=True), f, -1)

    def load_params(self, filename):
        """Load parameters from a file."""
        with open(filename) as f:
            for param in self.params:
                param.set_value(cPickle.load(f), borrow=True)

class Layer:
    """Base class for all layers."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, dim_in, dim_out, *args, **kw):
        """The initialize function should provide the field `params`."""

    @abc.abstractmethod
    def y(self, x):
        """Given input `x`, provide the output `y`."""

class LogisticRegression(Classifier):
    def __init__(self, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.W = theano.shared(
            value = np.zeros((dim_in, dim_out), dtype = theano.config.floatX),
            name = 'W', borrow = True)
        self.b = theano.shared(
            value = np.zeros((dim_out,), dtype = theano.config.floatX),
            name = 'b', borrow = True)
        self.params = [self.W, self.b]

    def p_y_given_x(self, x):
        return T.nnet.softmax(T.dot(x, self.W) + self.b)

class HiddenLayer(Layer):
    def __init__(self, dim_in, dim_out, rng, W=None, b=None,
                 activation=T.tanh):
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low = -math.sqrt(6. / (dim_in + dim_out)),
                    high = math.sqrt(6. / (dim_in + dim_out)),
                    size = (dim_in, dim_out)),
                dtype = theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value = W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((dim_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.activation = activation

    def y(self, x):
        return self.activation(T.dot(x, self.W) + self.b)

class MultiLayerPerceptron(Classifier):
    def __init__(self, dim_in, dim_out, dim_hidden, rng):
        self.hiddenLayer = HiddenLayer(
            dim_in=dim_in, dim_out=dim_hidden, rng=rng,
            activation = T.tanh)
        self.logisticRegressionLayer = LogisticRegression(
            dim_in = dim_hidden, dim_out = dim_out)
        self.L1 = abs(self.hiddenLayer.W).sum() + \
                  abs(self.logisticRegressionLayer.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + \
                      (self.logisticRegressionLayer.W ** 2).sum()
        self.params = self.hiddenLayer.params + \
                      self.logisticRegressionLayer.params

    def p_y_given_x(self, x):
        return self.logisticRegressionLayer.p_y_given_x(
            self.hiddenLayer.y(x))

class LeNetConvPoolLayer(Layer):
    def __init__(self, image_shape, filter_shape, pool_size, rng):
        """
        image_shape = (batch_size, num_x_channels, x_height, x_width)
        filter_shape = (num_filters, num_x_channels,
                        filter_height, filter_width)
        pool_size = downsampling factor of (#rows, #height)
        """
        assert image_shape[1] == filter_shape[1]
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.pool_size = pool_size

        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:]) / \
                  np.prod(pool_size)
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W_values = np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX)
        b_values = np.zeros((filter_shape[0],), dtype= theano.config.floatX)
        self.W = theano.shared(value=W_values, borrow=True)
        self.b = theano.shared(value=b_values, borrow=True)
        self.params = [self.W, self.b]

    def y(self, x):
        conv_out = theano.tensor.nnet.conv.conv2d(
            input=x,
            filters = self.W,
            filter_shape = self.filter_shape,
            image_shape = self.image_shape)
        pooled_out = theano.tensor.signal.downsample.max_pool_2d(
            input=conv_out, ds=self.pool_size, ignore_border=True)
        return T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class LeNet5(Classifier):
    def __init__(self, dim_in, dim_out, num_kernels, rng):
        """
        dim_in = (batch_size, num_x_channels, x_height, x_width)
        num_kernels = (num_kernels0, num_kernels1)
        """
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.layer0 = LeNetConvPoolLayer(
            filter_shape = (num_kernels[0], 1, 5, 5),
            image_shape = dim_in,
            pool_size = (2, 2),
            rng = rng)
        layer1_dim_in = (dim_in[0], num_kernels[0],
                         (dim_in[2]-5+1)/2, (dim_in[2]-5+1)/2)
        self.layer1 = LeNetConvPoolLayer(
            filter_shape = (num_kernels[1], num_kernels[0], 5, 5),
            image_shape = layer1_dim_in,
            pool_size = (2, 2),
            rng = rng)
        self.layer2 = HiddenLayer(
            dim_in = num_kernels[1] * 4 * 4,
            dim_out = 500,
            rng = rng,
            activation = T.tanh)
        self.layer3 = LogisticRegression(
            dim_in = 500,
            dim_out = 10)
        self.params = self.layer0.params + self.layer1.params + \
                      self.layer2.params + self.layer3.params

    def p_y_given_x(self, x):
        return self.layer3.p_y_given_x(
            self.layer2.y(
                self.layer1.y(
                    self.layer0.y(x.reshape(self.dim_in))
                ).flatten(2)
            )
        )
