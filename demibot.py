import tensorflow as tf
import numpy as np

import seq2seq_wrapper
# preprocessed data
import data
import data_utils


# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 10
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

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

#TODO take user input for questions
'''
quest = input('Hi, how are you?')
quest = quest.lower()
quest = data.filter_line(quest, data.EN_WHITELIST)
que_tok = [w.strip() for w in quest.split(' ') if w]
for q in zip(que_tok):
    print('q : [{0}]'.format(q))
'''

input_ = test_batch_gen.__next__()[0]
output = model.predict(sess, input_)
print(output.shape)
replies = []
for ii, oi in zip(input_.T, output):
    q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
    decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
    if decoded.count('unk') == 0:
        if decoded not in replies:
            print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
            replies.append(decoded)

