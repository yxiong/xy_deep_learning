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

class FileIOBase:
    """An abstract base class provides simple File I/O utilities."""
    __metaclass__ = abc.ABCMeta

    @classmethod
    def load_from_file(cls, f):
        """Load the object from a file `f`, which can either be a filename
        specified by a string or a file-like object."""
        if isinstance(f, basestring):
            with open(f, 'rb') as file_obj:
                return cls.load_from_file_obj(file_obj)
        else:
            return cls.load_from_file_obj(f)

    @classmethod
    @abc.abstractmethod
    def load_from_file_obj(cls, f):
        """Load the classifier from a file-like object."""

    def save_to_file(self, f):
        """Save the object to a file `f`, which can be either a filename
        specified by a string or a file-like object."""
        if isinstance(f, basestring):
            with open(f, 'wb') as file_obj:
                self.save_to_file(file_obj)
        else:
            self.save_to_file_obj(f)

    @abc.abstractmethod
    def save_to_file_obj(self, f):
        """Save the classifier to a file-like object."""

class Classifier(FileIOBase):
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

class Layer(FileIOBase):
    """Base class for all layers."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, dim_in, dim_out, *args, **kw):
        """The initialize function should provide the field `params`."""

    @abc.abstractmethod
    def y(self, x):
        """Given input `x`, provide the output `y`."""

class LogisticRegression(Classifier):
    def __init__(self, dim_in, dim_out, W=None, b=None):
        self.dim_in = dim_in
        self.dim_out = dim_out
        if W is None:
            W = theano.shared(
                value = np.zeros((dim_in, dim_out), dtype=theano.config.floatX),
                name = 'W', borrow = True)
        if b is None:
            b = theano.shared(
                value = np.zeros((dim_out,), dtype=theano.config.floatX),
                name = 'b', borrow = True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]

    @classmethod
    def load_from_file_obj(cls, f):
        dim_in = cPickle.load(f)
        dim_out = cPickle.load(f)
        W = theano.shared(cPickle.load(f), name='W', borrow=True)
        b = theano.shared(cPickle.load(f), name='b', borrow=True)
        return cls(dim_in, dim_out, W, b)

    def save_to_file_obj(self, f):
        protocol = cPickle.HIGHEST_PROTOCOL
        cPickle.dump(self.dim_in, f, protocol)
        cPickle.dump(self.dim_out, f, protocol)
        cPickle.dump(self.W.get_value(borrow=True), f, protocol)
        cPickle.dump(self.b.get_value(borrow=True), f, protocol)

    def p_y_given_x(self, x):
        return T.nnet.softmax(T.dot(x, self.W) + self.b)

class HiddenLayer(Layer):
    def __init__(self, dim_in, dim_out, W=None, b=None,
                 activation=T.tanh, rng=None):
        self.dim_in = dim_in
        self.dim_out = dim_out
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

    @classmethod
    def load_from_file_obj(cls, f):
        dim_in = cPickle.load(f)
        dim_out = cPickle.load(f)
        W = theano.shared(cPickle.load(f), name='W', borrow=True)
        b = theano.shared(cPickle.load(f), name='b', borrow=True)
        activation = cls.activation_from_str(cPickle.load(f))
        return cls(dim_in, dim_out, W=W, b=b, activation=activation)

    @classmethod
    def activation_from_str(cls, activation_str):
        if activation_str == "tanh":
            return T.tanh
        elif activation_str == "sigmoid":
            return T.nnet.sigmoid
        else:
            raise Exception("Unknown activation string.")

    def save_to_file_obj(self, f):
        protocol = cPickle.HIGHEST_PROTOCOL
        cPickle.dump(self.dim_in, f, protocol)
        cPickle.dump(self.dim_out, f, protocol)
        cPickle.dump(self.W.get_value(borrow=True), f, protocol)
        cPickle.dump(self.b.get_value(borrow=True), f, protocol)
        cPickle.dump(self.activation_str(), f, protocol)

    def activation_str(self):
        if self.activation == T.tanh:
            return "tanh"
        elif self.activation == T.nnet.sigmoid:
            return "sigmoid"
        else:
            raise Exception("Unknown activation type.")

    def y(self, x):
        return self.activation(T.dot(x, self.W) + self.b)

class MultiLayerPerceptron(Classifier):
    def __init__(self, dim_in, dim_out, dim_hidden=500,
                 hiddenLayer = None, logisticRegressionLayer = None,
                 rng=None):
        self.dim_in = dim_in
        self.dim_out = dim_out
        if hiddenLayer is None:
            hiddenLayer = HiddenLayer(
                dim_in=dim_in, dim_out=dim_hidden, activation = T.tanh,
                rng=rng)
        if logisticRegressionLayer is None:
            logisticRegressionLayer = LogisticRegression(
                dim_in = dim_hidden, dim_out = dim_out)
        self.hiddenLayer = hiddenLayer
        self.logisticRegressionLayer = logisticRegressionLayer
        self.L1 = abs(self.hiddenLayer.W).sum() + \
                  abs(self.logisticRegressionLayer.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + \
                      (self.logisticRegressionLayer.W ** 2).sum()
        self.params = self.hiddenLayer.params + \
                      self.logisticRegressionLayer.params

    @classmethod
    def load_from_file_obj(cls, f):
        dim_in = cPickle.load(f)
        dim_out = cPickle.load(f)
        hiddenLayer = HiddenLayer.load_from_file_obj(f)
        logisticRegressionLayer = LogisticRegression.load_from_file_obj(f)
        return cls(dim_in, dim_out,
                   hiddenLayer=hiddenLayer,
                   logisticRegressionLayer = logisticRegressionLayer)

    def save_to_file_obj(self, f):
        protocol = cPickle.HIGHEST_PROTOCOL
        cPickle.dump(self.dim_in, f, protocol)
        cPickle.dump(self.dim_out, f, protocol)
        self.hiddenLayer.save_to_file_obj(f)
        self.logisticRegressionLayer.save_to_file_obj(f)

    def p_y_given_x(self, x):
        return self.logisticRegressionLayer.p_y_given_x(
            self.hiddenLayer.y(x))

class LeNetConvPoolLayer(Layer):
    def __init__(self, image_shape, filter_shape, pool_size,
                 W = None, b = None, rng=None):
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

        if W is None:
            fan_in = np.prod(filter_shape[1:])
            fan_out = filter_shape[0] * np.prod(filter_shape[2:]) / \
                      np.prod(pool_size)
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W_values = np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX)
            W = theano.shared(value=W_values, borrow=True)
        if b is None:
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]

    @classmethod
    def load_from_file_obj(cls, f):
        image_shape = cPickle.load(f)
        filter_shape = cPickle.load(f)
        pool_size = cPickle.load(f)
        W = theano.shared(cPickle.load(f), name='W', borrow=True)
        b = theano.shared(cPickle.load(f), name='b', borrow=True)
        return cls(image_shape, filter_shape, pool_size, W=W, b=b)

    def save_to_file_obj(self, f):
        protocol = cPickle.HIGHEST_PROTOCOL
        cPickle.dump(self.image_shape, f, protocol)
        cPickle.dump(self.filter_shape, f, protocol)
        cPickle.dump(self.pool_size, f, protocol)
        cPickle.dump(self.W.get_value(borrow=True), f, protocol)
        cPickle.dump(self.b.get_value(borrow=True), f, protocol)

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
    # NOTE: the current classifier only work on batch data.
    def __init__(self, dim_in, dim_out, num_kernels=[20,50], layers=None,
                 rng=None):
        """
        dim_in = (batch_size, num_x_channels, x_height, x_width)
        num_kernels = (num_kernels0, num_kernels1)
        """
        self.dim_in = dim_in
        self.dim_out = dim_out

        if layers is None:
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
                activation = T.tanh,
                rng = rng)
            self.layer3 = LogisticRegression(
                dim_in = 500,
                dim_out = 10)
        else:
            self.layer0, self.layer1, self.layer2, self.layer3 = layers
        self.params = self.layer0.params + self.layer1.params + \
                      self.layer2.params + self.layer3.params

    @classmethod
    def load_from_file_obj(cls, f):
        dim_in = cPickle.load(f)
        dim_out = cPickle.load(f)
        layer0 = LeNetConvPoolLayer.load_from_file_obj(f)
        layer1 = LeNetConvPoolLayer.load_from_file_obj(f)
        layer2 = HiddenLayer.load_from_file_obj(f)
        layer3 = LogisticRegression.load_from_file_obj(f)
        return LeNet5(dim_in, dim_out,
                      layers=(layer0, layer1, layer2, layer3))

    def save_to_file_obj(self, f):
        protocol = cPickle.HIGHEST_PROTOCOL
        cPickle.dump(self.dim_in, f, protocol)
        cPickle.dump(self.dim_out, f, protocol)
        self.layer0.save_to_file_obj(f)
        self.layer1.save_to_file_obj(f)
        self.layer2.save_to_file_obj(f)
        self.layer3.save_to_file_obj(f)

    def p_y_given_x(self, x):
        return self.layer3.p_y_given_x(
            self.layer2.y(
                self.layer1.y(
                    self.layer0.y(x.reshape(self.dim_in))
                ).flatten(2)
            )
        )
