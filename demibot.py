import tensorflow as tf
import numpy as np

# preprocessed data
from datasets.cornell_corpus import data
import data_utils


#Model created
model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )

#Batch generation
val_batch_gen = data_utils.rand_batch_gen(validX, validY, 10)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, batch_size)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)

#Training the model
#sess = model.train(train_batch_gen, val_batch_gen, model.restore_last_session())

#Predicting the answers
sess = model.restore_last_session()