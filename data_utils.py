import numpy as np
from random import sample



'''
 split data into train (75%), test (15%) and valid(10%)
    return tuple( (trainX, trainY), (testX,testY), (validX,validY) )
'''
def split_dataset(x, y, ratio = [0.7, 0.15, 0.15] ):
    # number of examples
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]


#random batch generator
def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T
