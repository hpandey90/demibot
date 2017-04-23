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

#Check if it is a training session or not
motive = input("\nIs it a Training Session ? y/n \n")
if motive=='y' or motive=='Y':
    #Training the model
    sess = model.train(train_batch_gen, val_batch_gen)
elif motive=='n' or motive=='N':
    #Predicting the answers
    #load the previously saved model
    sess = model.restore_last_session()
    #ask your question
    while True:
        quest = input("\nask your question :")
        quest = quest.lower()
        quest = Final_data.filter_line(quest, Final_data.EN_WHITELIST)
        que_tok = [w.strip() for w in quest.split(' ') if w]
        #for q in zip(que_tok):
        print(que_tok)
        inp_idx = Final_data.pad_seq(que_tok,metadata['w2idx'],Final_data.limit['maxq'])
        #for q in range(inp_idx):
        #print(inp_idx)
        inp_idx_arr = np.zeros([1, Final_data.limit['maxq']], dtype=np.int32)
        inp_idx_arr[0] = np.array(inp_idx)
        #print(inp_idx_arr.shape)
        input_ = test_batch_gen.__next__()[0]
        output = model.predict(sess, inp_idx_arr.T)

        #replies = []
        for ii, oi in zip(inp_idx_arr, output):
            q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
            decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
            print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
