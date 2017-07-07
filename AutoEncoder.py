import theano
import theano.tensor as T
import numpy
from logistic import LogisticRegression
import logistic
import six.moves.cPickle as pickle
import pandas as pd
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class dA(object):
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            input=None,
            n_visible=784,
            n_hidden=500,
            W=None,
            bhid=None,
            bvis=None
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)
        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstruct_input(self,hidden):
        return T.nnet.sigmoid(T.dot(hidden,self.W_prime)+self.b_prime)

    def get_cost_updates(self,corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x , corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstruct_input(y)
        L = -T.sum(self.x * T.log(z) + (1-self.x)*T.log(1-z),axis=1)
        cost = T.mean(L)
        gparams = T.grad(cost,self.params)
        updates = [
            (param,param - learning_rate*gparam)
            for param , gparam in zip(self.params,gparams)
        ]
        return (cost,updates)

def test_dA(learning_rate=0.1, training_epochs=15,batch_size=500, output_folder='dA_plots.csv'):
    print('... data prepare')
    dataset = logistic.load_my_train_data()
    train_set_x, train_set_y = dataset[0]
    test_set_x, test_set_y = dataset[1]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    print('... building the model')

    index = T.lscalar()
    x = T.matrix('x')
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    for epoch in range(training_epochs):
        c = []
        for batch_index in range(n_train_batches):
            print('raining epoch %d',batch_index)
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, numpy.mean(c, dtype='float64'))



if __name__ == '__main__':
    test_dA()