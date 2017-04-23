from tkinter import *
from tkinter.font import Font
from ttk import *
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox

window = Tk()

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