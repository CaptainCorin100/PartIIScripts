import cv2
from PIL import Image
import tkinter as tk
from tkinter import filedialog

def init():
    root = tk.Tk()
    root.title("Part II Scripts")
    root.geometry("350x200")

    line_button = tk.Button(root,text="Analyse Line Thickness", command=analyse_line_thickness, anchor="center")
    line_button.pack(padx=20,pady=20)
    
    root.mainloop()

def analyse_line_thickness():
    file_path = filedialog.askopenfilename(filetypes=[("JPEG images", "*.jpg")])
    print(file_path)

if __name__ == "__main__":
    init()
    