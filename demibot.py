import tensorflow as tf
import numpy as np

import seq2seq_wrapper
# preprocessed data
import data
import data_utils

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)
