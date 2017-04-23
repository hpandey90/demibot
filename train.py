import tensorflow as tf
import numpy as np

import seq2seq
# preprocessed data
import Final_data
import data_utils

# load data from pickle and npy files
metadata, idx_q, idx_a = Final_data.load_data(PATH='./Final_META/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 512
