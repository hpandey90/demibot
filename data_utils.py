import numpy as np
from random import sample


'''
 generate batches, by random sampling a bunch of items
    yield (x_gen, y_gen)

'''
def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T