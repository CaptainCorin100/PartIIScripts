import cv2
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import numpy as np

kernel_size = (15,15)
kernel_elem = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

def init():
    root = tk.Tk()
    root.title("Part II Scripts")
    root.geometry("350x200")

    line_button = tk.Button(root,text="Analyse Line Thickness", command=analyse_line_thickness, anchor="center")
    line_button.pack(padx=20,pady=20)
    
    root.mainloop()

def analyse_line_thickness():
    #Load image from file
    file_path = filedialog.askopenfilename(filetypes=[("JPEG images", "*.jpg")])
    print(file_path)

    #Turn image into black and white
    img = cv2.imread(file_path)
    cv2.waitKey(0)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.blur(grey_img, (3,3))
    th, threshold_img = cv2.threshold(blurred_img, 170, 255, cv2.THRESH_OTSU)

    #Process image to extract contours
    closing = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel_elem)
    edges = cv2.Canny(closing, 60, 200)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_elem)
    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #Select two largest contours
    sorted_contours = sorted(contours, key=len, reverse=True)
    line_contours = sorted_contours[:2] #Only the largest two contours are actually useful

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i,c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c,3,True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    drawing = np.zeros((edges.shape[0], edges.shape[1],3), dtype=np.uint8)
    for i in range(len(contours)):
        cv2.drawContours(drawing,contours_poly,i,(0,255,0))
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,255,0), 2)

    #h,w = threshold_img.shape[:2]
    #mask = np.zeros((h+2, w+2), np.uint8)
    #floodfill_img = threshold_img.copy()
    #cv2.floodFill(floodfill_img, mask, (0,0), 255)
    #inverted_img = cv2.bitwise_not(floodfill_img)
    #combined_img = threshold_img | inverted_img

    displayed_contours = cv2.drawContours(img,line_contours,-1, (0,255,0), 2, cv2.LINE_AA)
    
    #dilation = cv2.dilate(edges, kernel, iterations=1)
    #erosion = cv2.erode(dilation, kernel, iterations=1)
    cv2.waitKey(0)

    resized = cv2.resize(displayed_contours, (1228, 921))
    cv2.imshow("Binarised Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    init()
