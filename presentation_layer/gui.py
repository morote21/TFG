import tkinter
import tkinter as tk
from tkinter import filedialog


def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print("Selected:", filename)


window = tk.Tk()

window_width = window.winfo_screenwidth()
window_height = window.winfo_screenheight()

window.geometry("%dx%d" % (window_width, window_height))
window.title("Statistics generator")

# label = tk.Label(window, text="Statistics generator")
# label.pack()


button = tk.Button(window, text="Open Video", command=UploadAction)
button.place(x=0, y=0)
window.mainloop()