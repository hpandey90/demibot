import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import seq2seq_wrapper

i = int(input('\n How many iterations ? \n'),10)
maxPerp = float(input('\n Max-Value of Y-Axis to plot ? \n'))

epochs = genfromtxt(seq2seq_wrapper.epochF, delimiter=',')
perp = genfromtxt(seq2seq_wrapper.perpF, delimiter=',')
loss = []
for per in perp:
    loss.append(math.log(per,2))

red_patch = mpatches.Patch(color='red', label='Perplexity')
blue_patch = mpatches.Patch(color='blue', label='Loss')
plt.legend(handles=[red_patch,blue_patch])
plt.plot(epochs[:i], perp[:i],'r',epochs[:i], loss[:i],'b')
plt.axis([1, i, 0, maxPerp])
plt.xlabel('Epochs')
plt.ylabel('Perplexity & Loss')
plt.title('Plot 1')
plt.grid(True)
plt.show()
plt.savefig("perplexity_new.png")
