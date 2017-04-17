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
