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

window.mainloop()
