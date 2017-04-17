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
