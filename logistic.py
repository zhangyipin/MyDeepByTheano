import numpy
import theano
import theano.tensor as T
import os
import sys
import six.moves.cPickle as pickle
import gzip
import timeit
import pandas as pd
from sklearn.model_selection import train_test_split
theano.config.floatX = 'float32'
class LogisticRegression(object):


    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in,n_out),
                dtype=theano.config.floatX
            )
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            )
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b)
        self.y_pred = T.argmax(self.p_y_given_x,axis=1)
        self.params = [self.W,self.b]
        self.input = input
    def negative_log_likehood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred,y))
        else:
            raise NotImplementedError()
def load_my_test_data(test_path = '../data/test.csv',borrow=True):
    test_set_sub = pd.read_csv(test_path)
    print test_set_sub.shape
    test_set_sub[test_set_sub > 0] = 1
    test_set_subm = theano.shared(numpy.asarray(test_set_sub, dtype=theano.config.floatX), borrow=borrow)
    return test_set_subm


def load_my_train_data(train_path='../data/train.csv',borrow=True):
    #train_path = '../data/train.csv'
    #test_path = '../data/test.csv'
    train_set = pd.read_csv(train_path)
    train_set = train_set[0:100]
    #train_set.to_csv('submission.csv', index=False, index_label=True)
    set_x = train_set.iloc[:,1:]
    set_x[numpy.abs(train_set) > 0] = 1
    set_y = train_set.iloc[:, 0:1].sum(axis=1)
   # print set_x
    print set_x.shape
    print set_y.shape
    #print set_y
    train_set_x,test_set_x, train_set_y,  test_set_y = train_test_split(set_x.values, set_y.values, test_size=0.2,
                                                        random_state=0)
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    train_set_x_x= theano.shared(numpy.asarray(train_set_x, dtype=theano.config.floatX), borrow=borrow)
    train_set_y_y = T.cast(theano.shared(numpy.asarray(train_set_y, dtype=theano.config.floatX), borrow=borrow),'int32')

    test_set_x_x = theano.shared(numpy.asarray(test_set_x, dtype=theano.config.floatX), borrow=borrow)
    test_set_y_y =  T.cast(theano.shared(numpy.asarray(test_set_y, dtype=theano.config.floatX), borrow=borrow),'int32')
    rval = [(train_set_x_x,train_set_y_y),[test_set_x_x,test_set_y_y]]
    return rval


def sgd_optimization_mnist_my(learning_rate=0.03, n_epochs=1000,
                           batch_size=600):
    dataset = load_my_train_data()
    train_set_x, train_set_y=dataset[0]
    test_set_x, test_set_y = dataset[1]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    cost = classifier.negative_log_likehood(y)
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    print('... training the model')
    done_looping = False
    epoch = 0
    best_test_score = 1
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            test_losses = [test_model(i) for i in range(n_test_batches)]
            test_score = numpy.mean(test_losses)
            print ('epoch %i testscore %f %%')% (epoch,test_score)

            if test_score < best_test_score:

                print(
                    (
                        '     epoch %i, minibatch %i/%i, test error of'
                        ' best model %f %%'
                    ) %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_score * 100.
                    )
                )
                best_test_score = test_score
                with open('best_model.pkl', 'wb') as f:
                    pickle.dump(classifier, f)


def predict():
    classifier = pickle.load(open('best_model.pkl'))
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)
    test_set = load_my_test_data()

    test_set_x = test_set.get_value()
    print type(test_set_x)
    print test_set_x.shape
    predicted_values = predict_model(test_set_x)
    print predicted_values.dtype
    ids = numpy.arange(predicted_values.shape[0]+1)
    print ids.dtype
    print predicted_values
    df = pd.DataFrame({"ImageId": ids[1:], "Label": predicted_values})
    print df
    df.to_csv('submission.csv', index=False, index_label=True)
    #sub_user_index['label'] = predicted_values
    #print("Predicted values for the first 10 examples in test set:")
    #print(predicted_values)


if __name__ == '__main__':
    #load_my_train_data()
    #load_my_test_data()
    sgd_optimization_mnist_my()
    predict()
    #dataset = 'mnist.pkl.gz'
    #datasets = load_data(dataset)
    #load_my_train_data()

    #print f([1,2,3,4])
