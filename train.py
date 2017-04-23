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

#Model created
model = seq2seq.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )

#Batch generation
val_batch_gen = data_utils.rand_batch_gen(validX, validY, batch_size)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, batch_size)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
