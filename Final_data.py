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
