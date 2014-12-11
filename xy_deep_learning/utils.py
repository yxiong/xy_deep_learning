#!/usr/bin/env python
#
# Author: Ying Xiong.
# Created: Dec 08, 2014.

import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
import time

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, "int32")

def load_mnist_data():
    with gzip.open("../data/mnist.pkl.gz", 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
    return train_set, valid_set, test_set

def create_minibatch_models(
        classifier, datasets, batch_size, learning_rate,
        x = T.matrix('x'), y = T.ivector('y'), cost = None):
    tr_set, vl_set, te_set = datasets
    tr_set_x, tr_set_y = shared_dataset(tr_set)
    vl_set_x, vl_set_y = shared_dataset(vl_set)
    te_set_x, te_set_y = shared_dataset(te_set)

    tr_num_batches = tr_set_x.get_value(borrow=True).shape[0] / batch_size
    vl_num_batches = vl_set_x.get_value(borrow=True).shape[0] / batch_size
    te_num_batches = te_set_x.get_value(borrow=True).shape[0] / batch_size
    num_batches = (tr_num_batches, vl_num_batches, te_num_batches)

    index = T.lscalar()
    if cost is None:
        cost = classifier.negative_log_likelihood(x, y)
    grads = [T.grad(cost, param) for param in classifier.params]
    updates = [(param, param - learning_rate * grad) \
               for param, grad in zip(classifier.params, grads)]
    tr_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: tr_set_x[index * batch_size : (index+1) * batch_size],
            y: tr_set_y[index * batch_size : (index+1) * batch_size]})
    vl_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(x, y),
        givens = {
            x: vl_set_x[index * batch_size : (index+1) * batch_size],
            y: vl_set_y[index * batch_size : (index+1) * batch_size]})
    te_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(x, y),
        givens = {
            x: te_set_x[index * batch_size : (index+1) * batch_size],
            y: te_set_y[index * batch_size : (index+1) * batch_size]})
    models = (tr_model, vl_model, te_model)

    return models, num_batches

def train_model(models, num_batches, num_epochs, patience):
    tr_model, vl_model, te_model = models
    tr_num_batches, vl_num_batches, te_num_batches = num_batches

    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(tr_num_batches, patience / 2)
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    iIter = -1
    while (epoch < num_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(tr_num_batches):
            iIter += 1
            tr_model(minibatch_index)
            if (iIter+1) % validation_frequency == 0:
                validation_losses = [vl_model(i)
                                     for i in xrange(vl_num_batches)]
                this_validation_loss = np.mean(validation_losses)
                print "epoch %i, minibatch %i/%i, validation error %f %%" % \
                    (epoch, minibatch_index+1, tr_num_batches,
                     this_validation_loss * 100)
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * \
                       improvement_threshold:
                        patience = max(patience, iIter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iIter
                    test_losses = [te_model(i)
                                   for i in xrange(te_num_batches)]
                    test_score = np.mean(test_losses)
                    print ("    epoch %i, minibatch %i/%i, test error of "
                           "best model %f %%" % \
                           (epoch, minibatch_index+1, tr_num_batches,
                            test_score * 100))
            if patience <= iIter:
                done_looping = True
                break

    end_time = time.clock()
    print ("Training complete. Best validation score of %f %%, "
           "obtained at iteration %i, with test performance %f %%") % \
        (best_validation_loss * 100., best_iter+1, test_score * 100.)
    if end_time - start_time < 100:
        print "The code run for %d epochs, with %f epochs/sec" % \
            (epoch, float(epoch) / (end_time - start_time))
        print "The code ran for %.1fsec" % (end_time - start_time)
    else:
        print "The code run for %d epochs, with %f epochs/min" % \
            (epoch, float(epoch) / (end_time - start_time) * 60.)
        print "The code ran for %.1fmin" % ((end_time - start_time)/60.)
