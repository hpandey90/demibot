import tensorflow as tf
import numpy as np

import seq2seq
# preprocessed data

#import data
import Cornell_data
import data_utils


# load data from pickle and npy files
metadata, idx_q, idx_a = Cornell_data.load_data(PATH='./Cornell/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 10
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

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
val_batch_gen = data_utils.rand_batch_gen(validX, validY, 10)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, batch_size)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)

#Check if it is a training session or not
motive = input("Is it a Training Session ? y/n")
if motive=='y' or motive=='Y':
    #Training the model
    sess = model.train(train_batch_gen, val_batch_gen)
elif motive=='n' or motive=='N':
    #Predicting the answers
    #load the previously saved model
    sess = model.restore_last_session()
    #ask your question
    while True:
        quest = input("ask your question :")
        quest = quest.lower()
        quest = data.filter_line(quest, data.EN_WHITELIST)
        que_tok = [w.strip() for w in quest.split(' ') if w]
        #for q in zip(que_tok):
        print(que_tok)
        inp_idx = data.pad_seq(que_tok,metadata['w2idx'],data.limit['maxq'])
        #for q in range(inp_idx):
        #print(inp_idx)
        inp_idx_arr = np.zeros([1, data.limit['maxq']], dtype=np.int32)
        inp_idx_arr[0] = np.array(inp_idx)
        print(inp_idx_arr.shape)
        input_ = test_batch_gen.__next__()[0]
        output = model.predict(sess, inp_idx_arr.T)

        #replies = []
        for ii, oi in zip(inp_idx_arr, output):
            q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
            decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
            if decoded.count('unk') == 0:
                #if decoded not in replies:
                print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
                #replies.append(decoded)
            else :
                print('q : [{0}]; a : [My bad! could not find a good answer]'.format(q))
