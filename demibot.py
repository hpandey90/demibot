from tkinter import *
from tkinter.font import Font
from ttk import *
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox

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


window.mainloop()



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
    
    for q in zip(que_tok):
    print(que_tok)
    inp_idx = Final_data.pad_seq(que_tok,metadata['w2idx'],Final_data.limit['maxq'])
    for q in range(inp_idx):