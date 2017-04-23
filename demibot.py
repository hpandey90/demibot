from tkinter import *
from tkinter.font import Font
from ttk import *
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox

import tensorflow as tf
import numpy as np

import seq2seq_wrapper
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
val_batch_gen = data_utils.rand_batch_gen(validX, validY, batch_size)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, batch_size)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)

sess = model.restore_last_session()

#Create a window frame
window = Tk()
#Set Icon
window.iconbitmap(default='logo.ico')
#Set Size
window.geometry("500x600")
#Set Style
window.style = Style()
#('clam', 'alt', 'default', 'classic')
window.style.theme_use("clam")
#Give title to window
window.wm_title("demiBot")
window.option_add('*font', 'Helvetica 11')

#Menu Funtion Reset
def reset():
    messages.config(state=NORMAL)
    messages.delete('1.0', END)
    messages.config(state=DISABLED)

#Menu Funtion About
def about():
    messagebox.showinfo("About demiBot", "demiBot is a chatter bot trained on movie dialogues from over 3000 movies with over 290,000 dialouges. \n\nApart from being a movie aficionado it also has a little bit of twitter in him.\n\n\t\tEnjoy chatting with him!!!")

#Menubar
menubar = Menu(window)
menubar.add_command(label="Reset", command=reset)
menubar.add_command(label="About", command=about)
menubar.add_command(label="Quit!", command=window.quit)
window.config(menu=menubar)

#Message Box For Chat
messages = ScrolledText(window)
messages.pack(fill=X,padx=10, pady=10)
messages.config(state=DISABLED)
messages.tag_configure('tag-left', justify='left')
messages.tag_configure('tag-right', justify='right')
UserFont = Font(family="Malgun Gothic Semilight", size=12)

#Input Field For Typing
input_user = StringVar()
input_field = Entry(window, text=input_user)
input_field.pack(side=BOTTOM, fill=X, padx=15, pady=15)

def Enter_pressed(event):
    quest = input_field.get()
    messages.config(state=NORMAL)
    messages.configure(font=UserFont)

    messages.insert(INSERT, 'User : \n%s\n' % quest,'tag-left')
    messages.yview_moveto(messages.yview()[1])
    input_user.set('')
    print(quest)

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
    answ = ''
    for ii, oi in zip(inp_idx_arr, output):
        q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
        decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
        #if decoded.count('unk') == 0:
        #    if decoded not in replies:
        print('q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded)))
        answ = ' '.join(decoded)

    #messages.configure(font=demiFont)
    messages.insert(INSERT, 'demiBot : \n%s\n' % answ, 'tag-right')
    messages.yview_moveto(messages.yview()[1])
    messages.config(state=DISABLED)
    return "break"

#frame window with bind value
frame = Frame(window)  # , width=300, height=300)
input_field.bind("<Return>", Enter_pressed)
frame.pack()

window.mainloop()
