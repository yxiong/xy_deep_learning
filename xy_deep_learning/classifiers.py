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

    def classify_data(self, x_data):
        """Classify data `x`."""
        if not hasattr(self, "classify_fcn"):
            x = T.matrix('x')
            self.classify_fcn = theano.function(
                inputs = [x],
                outputs = self.classify(x))
        return self.classify_fcn(x_data)

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
        self.L1 = abs(self.W).sum()
        self.L2_sqr = (self.W ** 2).sum()
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
        self.L1 = abs(self.W).sum()
        self.L2_sqr = (self.W ** 2).sum()
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
    def __init__(self, dim_in, dim_out, dim_hiddens,
                 hidden_layers = None, logistic_regression_layer = None,
                 rng=None):
        """dim_hiddens: an array for the (output dimension) of hidden layers."""
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hiddens = dim_hiddens
        if hidden_layers is None:
            next_dim_in = dim_in
            hidden_layers = []
            for i in xrange(len(dim_hiddens)):
                hidden_layers.append(HiddenLayer(
                    dim_in=next_dim_in, dim_out=dim_hiddens[i],
                    activation=T.tanh, rng=rng))
                next_dim_in = dim_hiddens[i]
        if logistic_regression_layer is None:
            logistic_regression_layer = LogisticRegression(
                dim_in = next_dim_in, dim_out = dim_out)
        self.hidden_layers = hidden_layers
        self.logistic_regression_layer = logistic_regression_layer
        self.L1 = sum([l.L1 for l in self.hidden_layers]) + \
                  self.logistic_regression_layer.L1
        self.L2_sqr = sum([l.L2_sqr for l in self.hidden_layers]) + \
                      self.logistic_regression_layer.L2_sqr
        self.params = sum([l.params for l in self.hidden_layers], []) + \
                      self.logistic_regression_layer.params

    @classmethod
    def load_from_file_obj(cls, f):
        dim_in = cPickle.load(f)
        dim_out = cPickle.load(f)
        dim_hiddens = cPickle.load(f)
        hidden_layers = []
        for i in xrange(len(dim_hiddens)):
            hidden_layers.append(HiddenLayer.load_from_file_obj(f))
        logistic_regression_layer = LogisticRegression.load_from_file_obj(f)
        return cls(dim_in, dim_out, dim_hiddens = dim_hiddens,
                   hidden_layers = hidden_layers,
                   logistic_regression_layer = logistic_regression_layer)

    def save_to_file_obj(self, f):
        protocol = cPickle.HIGHEST_PROTOCOL
        cPickle.dump(self.dim_in, f, protocol)
        cPickle.dump(self.dim_out, f, protocol)
        cPickle.dump(self.dim_hiddens, f, protocol)
        for hidden_layer in self.hidden_layers:
            hidden_layer.save_to_file_obj(f)
        self.logistic_regression_layer.save_to_file_obj(f)

    def p_y_given_x(self, x):
        next_in = x
        for hidden_layer in self.hidden_layers:
            next_out = hidden_layer.y(next_in)
            next_in = next_out
        return self.logistic_regression_layer.p_y_given_x(next_in)

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
        self.dim_out = [image_shape[0], filter_shape[0],
                        (image_shape[2]-filter_shape[2]+1) / pool_size[0],
                        (image_shape[3]-filter_shape[3]+1) / pool_size[1]]

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

class LeNet(Classifier):
    def __init__(self, dim_in, dim_out, batch_size,
                 dim_convs, dim_pools, dim_hiddens,
                 conv_pool_layers=None, multi_layer_perceptron=None, rng=None):
        """
        dim_in: 3-tuple of form (num_channels, height, width)
        dim_convs: a list of 3-tuples of form
                   (num_kernels, filter_height, filter_width)
        dim_pools: a list of 2-tuples of form (pool_width, pool_height).
        dim_hiddens: a list of integers for output dimension of hidden layers.
        """
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.batch_size = batch_size
        self.dim_convs = dim_convs
        self.dim_pools = dim_pools
        self.dim_hiddens = dim_hiddens
        if conv_pool_layers is None:
            conv_pool_layers = []
            next_dim_in = [batch_size, dim_in[0], dim_in[1], dim_in[2]]
            for dim_conv, dim_pool in zip(dim_convs, dim_pools):
                filter_shape = (dim_conv[0], next_dim_in[1],
                                dim_conv[1], dim_conv[2])
                conv_pool_layers.append(LeNetConvPoolLayer(
                    filter_shape = filter_shape,
                    image_shape = next_dim_in,
                    pool_size = dim_pool,
                    rng = rng))
                next_dim_in = conv_pool_layers[-1].dim_out
        if multi_layer_perceptron is None:
            multi_layer_perceptron = MultiLayerPerceptron(
                dim_in = next_dim_in[1] * next_dim_in[2] * next_dim_in[3],
                dim_out = dim_out, dim_hiddens = dim_hiddens, rng=rng)
        self.conv_pool_layers = conv_pool_layers
        self.multi_layer_perceptron = multi_layer_perceptron
        self.params = sum([l.params for l in self.conv_pool_layers], []) + \
                      self.multi_layer_perceptron.params

    @classmethod
    def load_from_file_obj(cls, f):
        dim_in = cPickle.load(f)
        dim_out = cPickle.load(f)
        batch_size = cPickle.load(f)
        dim_convs = cPickle.load(f)
        dim_pools = cPickle.load(f)
        dim_hiddens = cPickle.load(f)
        conv_pool_layers = []
        for i in xrange(len(dim_convs)):
            conv_pool_layers.append(LeNetConvPoolLayer.load_from_file_obj(f))
        multi_layer_perceptron = MultiLayerPerceptron.load_from_file_obj(f)
        return cls(dim_in, dim_out, batch_size,
                   dim_convs, dim_pools, dim_hiddens,
                   conv_pool_layers = conv_pool_layers,
                   multi_layer_perceptron = multi_layer_perceptron)

    def save_to_file_obj(self, f):
        protocol = cPickle.HIGHEST_PROTOCOL
        cPickle.dump(self.dim_in, f, protocol)
        cPickle.dump(self.dim_out, f, protocol)
        cPickle.dump(self.batch_size, f, protocol)
        cPickle.dump(self.dim_convs, f, protocol)
        cPickle.dump(self.dim_pools, f, protocol)
        cPickle.dump(self.dim_hiddens, f, protocol)
        for conv_pool_layer in self.conv_pool_layers:
            conv_pool_layer.save_to_file_obj(f)
        self.multi_layer_perceptron.save_to_file_obj(f)

    def p_y_given_x(self, x):
        next_in = x.reshape((self.batch_size, self.dim_in[0],
                             self.dim_in[1], self.dim_in[2]))
        for conv_pool_layer in self.conv_pool_layers:
            next_out = conv_pool_layer.y(next_in)
            next_in = next_out
        return self.multi_layer_perceptron.p_y_given_x(next_in.flatten(2))

    def classify_data(self, x_data):
        """We need to override this function because the symbolic `classify` only
        works on batch data."""
        if not hasattr(self, "classify_fcn"):
            x = T.matrix('x')
            self.classify_fcn = theano.function(
                inputs = [x],
                outputs = self.classify(x))
        # Process full batches.
        num_batches = x_data.shape[0] / self.batch_size
        y_batches = []
        for index in xrange(num_batches):
            y_batches.append(self.classify_fcn(x_data[
                index * self.batch_size : (index+1) * self.batch_size]))
        # Process remaining data that does not make a full batch.
        if x_data.shape[0] % self.batch_size > 0:
            data = x_data[num_batches * self.batch_size:, :]
            num_pad = self.batch_size - data.shape[0]
            pad = np.zeros((num_pad, x_data.shape[1]),
                           dtype=theano.config.floatX)
            x_batch = np.vstack((data, pad))
            y_batches.append(self.classify_fcn(x_batch)[:data.shape[0]])
        return np.concatenate(y_batches)
