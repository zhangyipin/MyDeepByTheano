import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from MLP import HiddenLayer
from logistic import LogisticRegression
import logistic
import six.moves.cPickle as pickle
import numpy
import pandas as pd
theano.config.floatX = 'float32'


rng = numpy.random.RandomState(23455)

class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.input = input
class CNN_data(object):
    def __init__(self,input):
        self.layer0 = pickle.load(open('cnnlayer0.pkl'))
        self.layer1 = pickle.load(open('cnnlayer1.pkl'))
        self.layer2 = pickle.load(open('cnnlayer2.pkl'))
        self.logRegressionLayer = pickle.load(open('cnnlayer3.pkl'))
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likehood
        )
        self.errors = self.logRegressionLayer.errors
        self.params = self.layer0.params + self.layer1.params + self.layer2.params + self.logRegressionLayer.params
        self.input = input
class CNN(object):
    def __init__(self, rng, input, n_hidden_out, n_out,nkerns,batch_size):

        self.layer0 = LeNetConvPoolLayer(
            rng,
            input= input.reshape((batch_size, 1, 28, 28)),
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )

        self.layer1 = LeNetConvPoolLayer(
            rng,
            input=self.layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0] , 5, 5),
            poolsize=(2, 2)
        )
        self.layer2 = HiddenLayer(
            rng,
            input=self.layer1.output.flatten(2),
            n_in=nkerns[1] * 4 * 4,
            n_out=n_hidden_out,
            activation=T.tanh
        )
        self.logRegressionLayer = LogisticRegression(
            input = self.layer2.output,
            n_in = n_hidden_out,
            n_out = n_out
        )
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likehood
        )

        self.errors = self.logRegressionLayer.errors

        self.params = self.layer0.params + self.layer1.params + self.layer2.params + self.logRegressionLayer.params

        self.input = input

def test_cnn(nkerns=(6,12),learning_rate=0.01,n_epochs=10,
              batch_size=60, n_hidden=500,n_out=10):
    print('... data prepare')
    dataset = logistic.load_my_train_data()
    train_set_x, train_set_y = dataset[0]
    test_set_x, test_set_y = dataset[1]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    print('... building the model')
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    rng = numpy.random.RandomState(1234)
   # layer0_input = x.reshape((batch_size, 1, 28, 28))
    cnn = CNN(
        rng=rng,
        input=x,
        n_hidden_out = n_hidden,
        n_out = n_out,
        nkerns = nkerns,
        batch_size=batch_size
    )
    cost = (
        cnn.negative_log_likelihood(y)
    )
    test_model = theano.function(
        inputs=[index],
        outputs=cnn.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    gparams = [T.grad(cost, param) for param in cnn.params]
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(cnn.params, gparams)
        ]
    train_model = theano.function(
        [index],
        cost,
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
            print minibatch_avg_cost
            test_losses = [test_model(i) for i in range(n_test_batches)]
            test_score = numpy.mean(test_losses)
            print ('epoch %i testscore %f %%') % (epoch, test_score)
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
                with open('cnnlayer0.pkl', 'wb') as f:
                    pickle.dump(cnn.layer0, f)
                with open('cnnlayer1.pkl', 'wb') as f:
                    pickle.dump(cnn.layer1, f)
                with open('cnnlayer2.pkl', 'wb') as f:
                    pickle.dump(cnn.layer2, f)
                with open('cnnlayer3.pkl', 'wb') as f:
                    pickle.dump(cnn.logRegressionLayer, f)
def predict():
    cnnlayer0 = pickle.load(open('cnnlayer0.pkl'))
    cnnlayer1 = pickle.load(open('cnnlayer1.pkl'))
    cnnlayer2 = pickle.load(open('cnnlayer2.pkl'))
    cnnlayer3 = pickle.load(open('cnnlayer3.pkl'))

    lout0 = theano.function(
        inputs=[cnnlayer0.input],
        outputs=cnnlayer0.output
    )
    lout1 = theano.function(
        inputs=[cnnlayer1.input],
        outputs=cnnlayer1.output
    )
    lout2 = theano.function(
        inputs=[cnnlayer2.input],
        outputs=cnnlayer2.output
    )
    lout3 = theano.function(
        inputs=[cnnlayer3.input],
        outputs=cnnlayer3.y_pred
    )

    test_set = logistic.load_my_test_data()
    test_set_x = test_set.get_value()
    #lout0_v = loutEnd(test_set_x)
    #lout1_v = lout1(test_set_x)
    #print test_set_x.shape
    test_set_x = test_set_x.reshape(28000L, 1, 28, 28)
    print test_set_x.shape
    #print type(test_set_x)
    test = T.lmatrix(name='test')
    lout0_v = lout0(test_set_x)
    lout1_v = lout1(lout0_v)
    lout_1_vv =lout1_v.reshape(28000,640)
    lout2_v = lout2(lout_1_vv)

   # predicted_value = lout3(lout2_v)

    predicted_values = lout3(lout2_v)
    #print predicted_values.dtype
    ids = numpy.arange(predicted_values.shape[0]+1)
    #print ids.dtype
   # print predicted_values
    df = pd.DataFrame({"ImageId": ids[1:], "Label": predicted_values})
    #print df
    df.to_csv('submission.csv', index=False, index_label=True)

if __name__ == '__main__':
    test_cnn()
    #predict()


