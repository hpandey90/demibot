UNK = 'unk'
VOCAB_SIZE = 8000
limit = {
        'maxq' : 25,
        'minq' : 2,
        'maxa' : 25,
        'mina' : 2
        }

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space included

FILENAME = 'data/raw_data/twitter/chat.txt'


import random
import sys

import nltk
import itertools
from collections import defaultdict

import numpy as np

import pickle


# read lines from file returns [list of lines]
def read_lines(filename):
    return open(filename,encoding='utf-8').read().split('\n')[:-1]


def split_line(line):
    return line.split('.')

#Filter according to the white list
def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''
def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences)//2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i+1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i+1])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


'''
 create the final dataset :
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )

'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a

#Padding the sequence to max length
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))
