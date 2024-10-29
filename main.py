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

    img = cv2.imread(file_path)
    cv2.waitKey(0)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshold_img = cv2.threshold(grey_img, 120, 255, cv2.THRESH_OTSU)
    edges = cv2.Canny(threshold_img, 60, 200)
    cv2.waitKey(0)

    resized = cv2.resize(edges, (1228, 921))
    cv2.imshow("Binarised Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    init()
