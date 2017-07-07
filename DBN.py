import theano
import theano.tensor as T
import numpy
from logistic import LogisticRegression
import logistic
import os
from RBM import RBM
from MLP import HiddenLayer
import pandas as pd
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
class DBN(object):
    def __init__(self,numpy_rng,theano_rng=None,n_ins=784,hidden_layers_sizes=[500,500],n_outs=10):
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = theano_rng = RandomStreams(numpy_rng.randint(2**30))

        self.x = T.matrix('x')
        self.y = T.ivector('y')
        for i in range(self.n_layers):
            if i==0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i-1]
            if i==0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output


            sigmoid_layer = HiddenLayer(
                rng=numpy_rng,
                input = layer_input,
                n_in=input_size,
                n_out = hidden_layers_sizes[i],
                activation=T.nnet.sigmoid
            )
            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)

            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        self.logLayer = LogisticRegression(
            input = self.sigmoid_layers[-1].output,
            n_in = hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)

        self.finetune_cost = self.logLayer.negative_log_likehood(self.y)

        self.errors = self.logLayer.errors(self.y)
    def pretraining_functions(self, train_set_x, batch_size, k):
        index = T.lscalar('index')
        learning_rate = T.scalar('lr')
        batch_begin = index*batch_size
        batch_end = batch_begin+batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:
            cost,updates = rbm.get_cost_updates(learning_rate,persistent=None,k=k)
            fn = theano.function(
                inputs = [index,theano.In(learning_rate,value=0.1)],
                outputs = cost,
                updates = updates,
                givens={
                    self.x:train_set_x[batch_begin:batch_end]
                }

            )
            pretrain_fns.append(fn)
        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        (train_set_x, train_set_y) = datasets[0]
        (test_set_x, test_set_y) = datasets[1]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size
        index = T.lscalar('index')  # index to a [mini]batch
        gparams = T.grad(self.finetune_cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))
        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                        index * batch_size: (index + 1) * batch_size
                        ],
                self.y: train_set_y[
                        index * batch_size: (index + 1) * batch_size
                        ]
            }
        )
        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                        index * batch_size: (index + 1) * batch_size
                        ],
                self.y: test_set_y[
                        index * batch_size: (index + 1) * batch_size
                        ]
            }
        )
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        valid_score = None
        return train_fn, valid_score, test_score

    def predict(self):
        test_set = logistic.load_my_test_data()
        test_set_x = test_set.get_value()
        predict_model = theano.function(
            inputs=[self.x],
            outputs=self.logLayer.y_pred)
        predicted_values = predict_model(test_set_x)
        ids = numpy.arange(predicted_values.shape[0] + 1)
        print ids.dtype
        print predicted_values
        df = pd.DataFrame({"ImageId": ids[1:], "Label": predicted_values})
        print df
        df.to_csv('submission.csv', index=False, index_label=True)


def test_DBN(finetune_lr=0.1, pretraining_epochs=10,
             pretrain_lr=0.01, k=1, training_epochs=10,
             dataset='mnist.pkl.gz', batch_size=10):

    datasets = logistic.load_my_train_data()

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    numpy_rng = numpy.random.RandomState(123)
    print('... building the model')
    dbn = DBN(numpy_rng=numpy_rng, n_ins=28 * 28,
              hidden_layers_sizes=[1000, 1000, 1000],
              n_outs=10)
    print('... getting the pretraining functions')

    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)
    print('... pre-training the model')

    for i in range(dbn.n_layers):
        for epoch in range(pretraining_epochs):
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost ' % (i, epoch))
            print(numpy.mean(c, dtype='float64'))
    print('The pretraining code for file ' + os.path.split(__file__)[1])
    train_fn,valid_model,test_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    print '.....finetuning the model'

    done_looping = False
    epoch = 0
    best_test_score = 1
    while (epoch < training_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            thefinetune_cost = train_fn(minibatch_index)
            print 'epoch %i, %f'% (epoch,thefinetune_cost)
            test_losses = test_model()
            test_score = numpy.mean(test_losses, dtype='float64')
            if test_score < best_test_score:
                print(('     epoch %i, minibatch %i/%i, test error of '
                       'best model %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches,
                              test_score * 100.))
                best_test_score = test_score
    print(('Optimization complete with best score of %f %%, '
           ) % (best_test_score * 100.))
    print 'start predict'
    dbn.predict()
if __name__ == '__main__':
    test_DBN()
    #predict()
