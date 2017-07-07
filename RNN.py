import theano
import theano.tensor as T
import numpy
import os
import sys
import copy
import random
import urllib
import gzip
import stat
import subprocess
from collections import OrderedDict
import six.moves.cPickle as pickle
import timeit
theano.config.floatX = 'float32'

PREFIX = os.getenv(
    'ATISDATA',
    os.path.join(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0],
                 'data'))

def shuffle(lol, seed):
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


def contextwin(l, win):
    assert (win % 2)==1
    assert win >= 1
    l = list(l)
    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]
    assert len(out) == len(l)
    return out

def atixfold(fold):
    assert fold in range(5)
    filename = os.path.join(PREFIX, 'atis.fold'+str(fold)+'.pkl.gz')
    f = gzip.open(filename, 'rb')
    try:
        train_set, valid_set, test_set, dicts = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set, dict = pickle.load(f)

    return train_set, valid_set, test_set, dict

def download(origin, destination):
    '''
    download the corresponding atis file
    from http://www-etud.iro.umontreal.ca/~mesnilgr/atis/
    '''
    print('Downloading data from %s' % origin)
    urllib.urlretrieve(origin, destination)

def conlleval(p, g, w, filename, script_path):
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    #return get_perf(filename, script_path)

def get_perf(filename, folder):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = os.path.join(folder, 'conlleval.pl')
    print "-----------------------------------"
    print _conlleval
    if not os.path.isfile(_conlleval):
        url = 'http://www-etud.iro.umontreal.ca/~mesnilgr/atis/conlleval.pl'
        download(url, _conlleval)
        os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                            _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()).encode('utf-8'))
    stdout = stdout.decode('utf-8')
    out = None

    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break
    # To help debug
    if out is None:
        print(stdout.split('\n'))
    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}
'''
run = RNNSLU(
        nh=param['nhidden'],
        nc = nclasses,
        ne = vocsize,
        de = param['emb_dimension'],
        cs = param['win']
    )
'''

class RNNSLU(object):
    def __init__(self, nh, nc, ne, de, cs):
        self.emb = theano.shared(name='embedding',
                                 value=0.2*numpy.random.uniform(-1.0,1.0,(ne+1,de))
                                 .astype(theano.config.floatX))

        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0,1.0,(de*cs,nh))
                                .astype(theano.config.floatX))

        self.wh = theano.shared(name='wh',
                                value=0.2*numpy.random.uniform(-1.0,1.0,(nh, nh))
                                .astype(theano.config.floatX))

        self.w = theano.shared(name='w',
                               value=0.2*numpy.random.uniform(-1.0,1.0,(nh, nc))
                               .astype(theano.config.floatX))

        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,dtype=theano.config.floatX))

        self.b = theano.shared(name='b',
                               value = numpy.zeros(nc,dtype=theano.config.floatX))

        self.h0 = theano.shared(name='h0',
                                value = numpy.zeros(nh,dtype=theano.config.floatX))

        self.params = [self.emb,self.wx,self.wh,self.w,self.bh,self.h0]

        idxs = T.imatrix()


        x = self.emb[idxs].reshape((idxs.shape[0],de*cs))

        y_sentence = T.ivector('y_sentence')

        def recurrence(x_t, h_tml):
            h_t = T.nnet.sigmoid(T.dot(x_t,self.wx)
                                 +T.dot(h_tml, self.wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)

            return [h_t, s_t]

        [h, s], _ = theano.scan(
            fn=recurrence,
            sequences=x,
            outputs_info=[self.h0,None],
            n_steps=x.shape[0]
        )

        p_y_given_x_sentence=s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence,axis=1)
        lr = T.scalar('lr')

        sentence_null = -T.mean(T.log(p_y_given_x_sentence)[T.arange(x.shape[0]),y_sentence])
        sentence_gradients = T.grad(sentence_null, self.params)
        sentence_updates = OrderedDict((p, p-lr*g)
                                       for p,g in zip(self.params,sentence_gradients))

        self.test = theano.function(inputs=[idxs], outputs=s)

        self.classify = theano.function(inputs=[idxs],outputs=y_pred)

        self.sentence_train = theano.function(inputs=[idxs,y_sentence,lr],
                                              outputs=sentence_null,
                                              updates=sentence_updates)
        self.normalize = theano.function(inputs=[],
                                          updates={self.emb:
                                                   self.emb/
                                                   T.sqrt((self.emb**2)
                                                    .sum(axis=1))
                                                    .dimshuffle(0,'x')})


    def train(self, x, y, window_size, learning_rate):
        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'),cwords))
        labels = y
        result = self.sentence_train(words,labels,learning_rate)
        self.normalize()
        return result
    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,param.name + '.npy'), param.get_value())


    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder, param.name + '.npy')))

def main(param = None):
    if not param:
        param = {
            'fold': 3,
            # 5 folds 0,1,2,3,4
            'data': 'atis',
            'lr': 0.0970806646812754,
            'verbose': 1,
            'decay': True,
            # decay on the learning rate if improvement stops
            'win': 7,
            # number of words in the context window
            'nhidden': 200,
            # number of hidden units
            'seed': 345,
            'emb_dimension': 50,
            # dimension of word embedding
            'nepochs': 10,
            # 60 is recommended
            'savemodel': False}


    folder_name = os.path.basename(__file__).split('.')[0]
    folder = os.path.join(os.path.dirname(__file__), folder_name)
    if not os.path.exists(folder):
        os.mkdir(folder)


    train_set, valid_set, test_set, dic = atixfold(param['fold'])

    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())

    idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex, test_ne, test_y = test_set
    #print len(valid_lex)
    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
    words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]
    groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
    words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

    numpy.random.seed(param['seed'])
    random.seed(param['seed'])
    rnn = RNNSLU(
        nh=param['nhidden'],
        nc = nclasses,
        ne = vocsize,
        de = param['emb_dimension'],
        cs = param['win']
    )

    best_f1 = -numpy.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):
        shuffle([train_lex, train_ne, train_y], param['seed'])
        param['ce'] = e
        tic = timeit.default_timer()
        for i, (x,y) in enumerate(zip(train_lex, train_y)):
            print ('error is:%f')%rnn.train(x, y, param['win'], param['clr'])

            print('[learning] epoch %i >> %2.2f%%' % (e, (i + 1) * 100./nsentences))
            print('completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic))
            sys.stdout.flush()
        predictions_test = [map(lambda x: idx2label[x],
                                rnn.classify(numpy.asarray(
                                    contextwin(x, param['win'])).astype('int32')))
                            for x in test_lex]
        #print predictions_test
        predictions_valid = [map(lambda x: idx2label[x],
                                 rnn.classify(numpy.asarray(
                                     contextwin(x, param['win'])).astype('int32')))
                             for x in valid_lex]
        #res_test =
        '''
        conlleval(predictions_test,
                             groundtruth_test,
                             words_test,
                             folder + '/current.test.txt',
                             folder)

        #res_valid =
        conlleval(predictions_valid,
                              groundtruth_valid,
                              words_valid,
                              folder + '/current.valid.txt',
                              folder)



        if param['savemodel']:
            rnn.save(folder)

        best_rnn = copy.deepcopy(rnn)
    '''
    print "end................................................................................"
    out = ''


    idx2word[572]="UNKNOW"
    print (idx2word)
    print rnn.emb.get_value().shape
    for index,sl in zip(numpy.arange(rnn.emb.get_value().shape[0]),rnn.emb.get_value()):
        #print index
        print sl
        out += '' + idx2word[index] + ":"
        for p in sl:
            print str(p)
            print p
            out+=str(p)+','
        out += "\n"
        #print index
        #out += ''+idx2word[index]+":"
        #out += sl.tostring()+"\n"
        #print sl.tostring()
        #print out
    f = open("result.txt", 'w')
    f.writelines(out)
    print "end write"
    f.close()

if __name__ == '__main__':
    #win = 7
    #l = [1,2,3,4,5,6,7,8,9]
    #lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    #print lpadded
    #out = [lpadded[i:(i + win)] for i in range(len(l))]
    #print out
    main()
'''
    train_set, valid_set, test_set, dic = atixfold(3)

    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())

    idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    train_lex, train_ne, train_y = train_set
    nclasses = len(dic['labels2idx'])
    vocsize = len(dic['words2idx'])
    rnn = RNNSLU(
        nh=200,
        nc=nclasses,
        ne=vocsize,
        de=5,
        cs=7
    )
    w = theano.shared(name='w',
                       value=0.2 * numpy.random.uniform(-1.0, 1.0, (7, 127))
                       .astype(theano.config.floatX))
    b = theano.shared(name='b',
                      value=numpy.zeros(127, dtype=theano.config.floatX))
    h_t = T.imatrix('h_t')
    s_t = T.nnet.softmax(T.dot(h_t, w) + b)


    fun = theano.function(
        inputs = [h_t],
        outputs = s_t
    )


    def recurrence(h_t):
        s_t = T.nnet.softmax(T.dot(h_t, w) + b)
        return [s_t]

    for i, (x, y) in enumerate(zip(train_lex, train_y)):
        cwords = contextwin(x, 7)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        newwords = numpy.asarray(words)
        print rnn.test(newwords).shape
        print fun(newwords).shape
        print newwords.shape
        print rnn.sentence_train(words,y,0.05)
'''
    #train_lex, train_ne, train_y = train_set
    #for i,j,k in zip(train_lex, train_ne, train_y ):
    #    print i
    #    print j
    #    print k
    #valid_lex, valid_ne, valid_y = valid_set
    #idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    #groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
    #dici = {v: k for k, v in dic['labels2idx'].items()}
    #print dici
    #_conlleval = "/mlProject/MyDeepByTheano/DigitRecognized/src\RNN\conlleval.pl"
    #proc = subprocess.Popen(["perl",
    #                         _conlleval],
    #                        stdin=subprocess.PIPE,
    #                        stdout=subprocess.PIPE)
    #best_f1 = -numpy.inf
    #print best_f1
    #dict = {"a": "apple", "b": "banana", "g": "grape", "o": "orange"}
    #y = ['a','b','a']
    #print [map(lambda x:dict[x],w) for w in y]



'''
{0: 'B-aircraft_code', 1: 'B-airline_code', 2: 'B-airline_name', 3: 'B-airport_code', 4: 'B-airport_name', 5: 'B-arrive_date.date_relative', 6: 'B-arrive_date.day_name', 7: 'B-arrive_date.day_number', 8: 'B-arrive_date.month_name', 9: 'B-arrive_date.today_relative', 10: 'B-arrive_time.end_time',
11: 'B-arrive_time.period_mod', 12: 'B-arrive_time.period_of_day', 13: 'B-arrive_time.start_time', 14: 'B-arrive_time.time', 15: 'B-arrive_time.time_relative', 16: 'B-booking_class', 17: 'B-city_name', 18: 'B-class_type', 19: 'B-compartment', 20: 'B-connect',
21: 'B-cost_relative', 22: 'B-day_name', 23: 'B-day_number', 24: 'B-days_code', 25: 'B-depart_date.date_relative', 26: 'B-depart_date.day_name', 27: 'B-depart_date.day_number', 28: 'B-depart_date.month_name', 29: 'B-depart_date.today_relative',
30: 'B-depart_date.year', 31: 'B-depart_time.end_time', 32: 'B-depart_time.period_mod', 33: 'B-depart_time.period_of_day', 34: 'B-depart_time.start_time', 35: 'B-depart_time.time', 36: 'B-depart_time.time_relative', 37: 'B-economy', 38: 'B-fare_amount', 39: 'B-fare_basis_code',
40: 'B-flight', 41: 'B-flight_days', 42: 'B-flight_mod', 43: 'B-flight_number', 44: 'B-flight_stop', 45: 'B-flight_time', 46: 'B-fromloc.airport_code', 47: 'B-fromloc.airport_name', 48: 'B-fromloc.city_name', 49: 'B-fromloc.state_code',
50: 'B-fromloc.state_name', 51: 'B-meal', 52: 'B-meal_code', 53: 'B-meal_description', 54: 'B-mod', 55: 'B-month_name', 56: 'B-or', 57: 'B-period_of_day', 58: 'B-restriction_code', 59: 'B-return_date.date_relative',
60: 'B-return_date.day_name', 61: 'B-return_date.day_number', 62: 'B-return_date.month_name', 63: 'B-return_date.today_relative', 64: 'B-return_time.period_mod', 65: 'B-return_time.period_of_day', 66: 'B-round_trip', 67: 'B-state_code', 68: 'B-state_name', 69: 'B-stoploc.airport_code',
70: 'B-stoploc.airport_name', 71: 'B-stoploc.city_name', 72: 'B-stoploc.state_code', 73: 'B-time', 74: 'B-time_relative', 75: 'B-today_relative', 76: 'B-toloc.airport_code', 77: 'B-toloc.airport_name', 78: 'B-toloc.city_name', 79: 'B-toloc.country_name',
80: 'B-toloc.state_code', 81: 'B-toloc.state_name', 82: 'B-transport_type', 83: 'I-airline_name', 84: 'I-airport_name', 85: 'I-arrive_date.day_number', 86: 'I-arrive_time.end_time', 87: 'I-arrive_time.period_of_day', 88: 'I-arrive_time.start_time', 89: 'I-arrive_time.time',
90: 'I-arrive_time.time_relative', 91: 'I-city_name', 92: 'I-class_type', 93: 'I-cost_relative', 94: 'I-depart_date.day_number', 95: 'I-depart_date.today_relative', 96: 'I-depart_time.end_time', 97: 'I-depart_time.period_of_day', 98: 'I-depart_time.start_time', 99: 'I-depart_time.time',
100: 'I-depart_time.time_relative', 101: 'I-economy', 102: 'I-fare_amount', 103: 'I-fare_basis_code', 104: 'I-flight_mod', 105: 'I-flight_number', 106: 'I-flight_stop', 107: 'I-flight_time', 108: 'I-fromloc.airport_name', 109: 'I-fromloc.city_name',
110: 'I-fromloc.state_name', 111: 'I-meal_code', 112: 'I-meal_description', 113: 'I-restriction_code', 114: 'I-return_date.date_relative', 115: 'I-return_date.day_number', 116: 'I-return_date.today_relative', 117: 'I-round_trip', 118: 'I-state_name', 119: 'I-stoploc.city_name',
120: 'I-time', 121: 'I-today_relative', 122: 'I-toloc.airport_name',123: 'I-toloc.city_name', 124: 'I-toloc.state_name', 125: 'I-transport_type', 126: 'O'}
'''




























































