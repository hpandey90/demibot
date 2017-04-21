import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import math

epochF = 'epoch.csv'
perpF = 'perp.csv'

class Seq2Seq(object):

    def __init__(self, xseq_len, yseq_len,
            xvocab_size, yvocab_size,
            emb_dim, num_layers, ckpt_path,
            lr=0.0001,
            epochs=100000, model_name='seq2seq_model'):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name
        self.xAxis = []
        self.yAxis = []
        self.lossAxis = []
        self.maxPerp = 0.0
        self.maxLoss = 0.0

        self.epochFile = open(epochF,"w", encoding='utf-8', errors='ignore')
        self.perpFile = open(perpF,"w", encoding='utf-8', errors='ignore')

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
            self.dec_ip = [tf.zeros_like(self.enc_ip[0],
                                         dtype=tf.int64, name='GO')]+self.labels[:-1]

            # Basic LSTM cell wrapped in Dropout Wrapper
            self.keep_prob = tf.placeholder(tf.float32)
            # define the basic cell
            basic_cell = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(emb_dim, state_is_tuple=True),
                output_keep_prob=self.keep_prob)
            # stack cells together : n layered model here n = 3
            stacked_lstm = tf.contrib.rnn.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)

            #Seq2Seq Model starts here
            with tf.variable_scope('decoder') as scope:
                # build the seq2seq model
                #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
                self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.enc_ip,self.dec_ip, stacked_lstm,
                                                    xvocab_size, yvocab_size, emb_dim)
                # share parameters
                scope.reuse_variables()
                # testing model, where output of previous timestep is fed as input
                #  to the next timestep
                self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                    feed_previous=True)

            # weighted loss function
            loss_weights = [ tf.ones_like(label, dtype=tf.float32) for label in self.labels ]
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, yvocab_size)
            # train op to minimize the loss
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        sys.stdout.write('<log> Building Graph ')
        # start the graph function here
        __graph__()
        sys.stdout.write('</log>')


    # getting the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob # dropout prob
        return feed_dict

    # run one batch for training
    def train_batch(self, sess, train_batch_gen):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=0.5)
        #get loss function
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)
        return loss_v

    #used as an intermediate function for eval_batches
    def eval_step(self, sess, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob=1.)
        loss_v, dec_op_v = sess.run([self.loss, self.decode_outputs_test], feed_dict)
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return loss_v, dec_op_v, batchX, batchY

    # evaluate 'num_batches' batches
    def eval_batches(self, sess, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)
        val_loss = np.mean(losses)
        val_perp = 2**val_loss
        return val_loss,val_perp

    def train(self, train_set, valid_set, sess=None ):

        # we need to save the model periodically
        saver = tf.train.Saver(max_to_keep=3)

        # if no session is given
        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())

        sys.stdout.write('\n<log> Training started </log>\n')
        for i in range(self.epochs):
            try:
                self.train_batch(sess, train_set)
                if i and i% (self.epochs//100) == 0: # TODO : make this tunable by the user
                    # save model to disk
                    saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                    # evaluate to get validation loss
                    val_loss,val_perp = self.eval_batches(sess, valid_set, 32) # TODO : and this
                    #self.xAxis.append(i)
                    self.epochFile.write('{0},'.format(i))
                    #self.yAxis.append(val_perp)
                    self.perpFile.write('{0:.6f}'.format(val_perp))
                    if self.maxPerp<val_perp:
                        self.maxPerp = val_perp
                    # print stats
                    print('\nModel saved to disk at iteration #{}'.format(i))
                    print('val   loss : {0:.6f}'.format(val_loss))
                    sys.stdout.flush()
                else:
                    val_loss,val_perp = self.eval_batches(sess, valid_set, 7)
                    #self.xAxis.append(i)
                    self.epochFile.write('{0},'.format(i))
                    #self.yAxis.append(val_perp)
                    self.perpFile.write('{0:.6f}'.format(val_perp))
                    if self.maxPerp<val_perp:
                        self.maxPerp = val_perp
            except KeyboardInterrupt: # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                self.session = sess
                return sess
        plt.plot(self.xAxis, self.yAxis)
        plt.axis([1, i, 0, self.maxPerp])
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity')
        plt.title('Plot 1')
        plt.grid(True)
        plt.savefig("perplexity.png")
        plt.show()

    def restore_last_session(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("\n Previous CheckPoint : "+ckpt.model_checkpoint_path)
        #return session variable to the user to restore last session
        return sess

    def predict(self, sess, X):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)
        #swap timestamp dimension and batch size
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)
