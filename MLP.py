import theano
import theano.tensor as T
import numpy
from logistic import LogisticRegression
import logistic
import six.moves.cPickle as pickle
import pandas as pd
theano.config.floatX = 'float32'

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,activation=T.tanh):
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low = -numpy.sqrt(6./(n_in + n_out)),
                    high= numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in,n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values,name='W',borrow=True)
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]
class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng = rng,
            input = input,
            n_in = n_in,
            n_out = n_hidden,
            activation = T.tanh
        )
        self.logRegressionLayer = LogisticRegression(
            input = self.hiddenLayer.output,
            n_in = n_hidden,
            n_out = n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()+abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()+(self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likehood
        )

        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input = input

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
              batch_size=600, n_hidden=500):
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

    classifier = MLP(
        rng = rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

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
                with open('best_modelh.pkl', 'wb') as f:
                    pickle.dump(classifier.hiddenLayer, f)
                with open('best_modell.pkl', 'wb') as f:
                    pickle.dump(classifier.logRegressionLayer, f)
def predict():
    classifierH = pickle.load(open('best_modelh.pkl'))
    classifierL = pickle.load(open('best_modell.pkl'))
    predict_modelH = theano.function(
        inputs=[classifierH.input],
        outputs=classifierH.output)

    predict_modelL = theano.function(
        inputs=[classifierL.input],
        outputs=classifierL.y_pred)

    test_set = logistic.load_my_test_data()
    test_set_x = test_set.get_value()

    predicted_valuesH = predict_modelH(test_set_x)
    predicted_values = predict_modelL(predicted_valuesH)

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
    #test_mlp()
    predict()

