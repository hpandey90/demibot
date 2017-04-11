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
