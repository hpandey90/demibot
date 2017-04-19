UNK = 'unk'
VOCAB_SIZE = 12000
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
import re
import glob, os

import numpy as np

import pickle

def get_file(filename):
    return open(filename,"a+", encoding='utf-8', errors='ignore')


def split_n_create(filename):
    lines = open(filename,encoding='utf-8').read().split('\n')[:-1]
    target = get_file(FILENAME)
    flag = 0
    counter = 0
    s = ""
    for line in lines:
        if line=="CUT TO:" or counter > 10:
            flag = 1
            counter = 0
        elif flag==1:
            _line = line.split(' ')
            if _line[0].isupper() and len(_line[0])!=1:
                if len(s)!=0:
                    tar = re.sub("[\(].*?[\)]", "", s)
                    tar = tar.strip()
                    target.write(tar)
                    target.write("\n")
                    s = ""
            elif len(line)!=0:
                s = ' '.join([s,line])
            counter = 0
        counter += 1


def many_to_one_file():
   for fl in glob.glob('./dialogs/**/*.txt'):
       split_n_create(fl)


def get_id2line():
    lines = open('./data/raw_data/Cornell/movie_lines.txt', encoding='utf-8', errors ='ignore').read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line


def get_conversations():
    conv = open('./data/raw_data/Cornell/movie_conversations.txt', encoding='utf-8', errors ='ignore').read().split('\n')
    convs = []
    for line in conv[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    return convs


def gather_dataset(convs, id2line):
    target = get_file(FILENAME)

    for conv in convs:
        if len(conv) %2 != 0:
            conv = conv[:-1]
        for i in range(len(conv)):
            if i%2 == 0:
                target.write(id2line[conv[i]])
                #questions.append(id2line[conv[i]])
            else:
                target.write(id2line[conv[i]])
                #answers.append(id2line[conv[i]])
            target.write('\n')


# read lines from file returns [list of lines]
def read_lines(filename):
    return open(filename,encoding='utf-8').read().split('\n')[:-1]


def split_line(line):
    return line.split('.')


#Filter according to the white list
def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])


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


def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences)//2
    print(len(sequences))

    for i in range(0, len(sequences)-1, 2):
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


def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

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

def process_data():

    print('\n>> Read lines from file')
    lines = read_lines(filename=FILENAME)

    # change to lower case (just for en)
    lines = [ line.lower() for line in lines ]
    print('\n:: Sample from read(p) lines')
    print(lines[121:125])
    # filter out unnecessary characters
    print('\n>> Filter lines')
    lines = [ filter_line(line, EN_WHITELIST) for line in lines ]
    print(lines[121:125])
    
    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')
    qlines, alines = filter_data(lines)
    print('\nq : {0} ; a : {1}'.format(qlines[60], alines[60]))
    print('\nq : {0} ; a : {1}'.format(qlines[61], alines[61]))
    
    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized = [ wordlist.split(' ') for wordlist in qlines ]
    atokenized = [ wordlist.split(' ') for wordlist in alines ]
    print('\n:: Sample from segmented list of words')
    print('\nq : {0} ; a : {1}'.format(qtokenized[60], atokenized[60]))
    print('\nq : {0} ; a : {1}'.format(qtokenized[61], atokenized[61]))