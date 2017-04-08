import tensorflow as tf
import numpy as np
import sys


class Seq2Seq(object):

    def __init__(self, xseq_len, yseq_len,
            xvocab_size, yvocab_size,
            emb_dim, num_layers, ckpt_path,
            lr=0.0001,
            epochs=100, model_name='seq2seq_model'):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name

        #graph building function
        def __graph__():

            # it is good to reset default graph before creating placeholders
            tf.reset_default_graph()
            #  encoder inputs : list of indices of length x sequence length
            self.enc_ip = [tf.placeholder(shape=[None,],
                                           dtype=tf.int64,
                                           name='ei_{}'.format(t)) for t in range(xseq_len)]

            #  labels that represent the real outputs
            self.labels = [tf.placeholder(shape=[None,],
                                          dtype=tf.int64,
                                          name='ei_{}'.format(t)) for t in range(yseq_len)]

            #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ] here Go is the separator
            #  between decoder and encoder
            self.dec_ip = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]

        sys.stdout.write('<log> Building Graph ')
        # start the graph function here
        __graph__()
        sys.stdout.write('</log>')



    # Training Code Starts

    # getting the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob # dropout prob = 0.5
        return feed_dict
