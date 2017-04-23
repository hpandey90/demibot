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