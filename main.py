import cv2
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import numpy as np

kernel_size = (15,15)
kernel_elem = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

threshold_cutoff = 170

sampling_rate = 2

green = (0,255,0)

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
    th, threshold_img = cv2.threshold(blurred_img, threshold_cutoff, 255, cv2.THRESH_OTSU)

    #Process image to extract contours
    closing = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel_elem)
    edges = cv2.Canny(closing, 60, 200)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_elem)
    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #Select two largest contours
    sorted_contours = sorted(contours, key=len, reverse=True)
    line_contours = sorted_contours[:2] #Only the largest two contours are actually useful

    drawing = img.copy()

    def converted_line_points (vx, vy, x, y):
        lefty = int((-x*vy/vx) + y)
        righty = int(((img.shape[1]-x)*vy/vx) + y)
        return [(img.shape[1] - 1, righty), (0, lefty)]

    #contours_poly = [None]*len(line_contours)
    #boundRect = [None]*len(contours)
    line_points = [None]*len(contours)

    for i,c in enumerate(line_contours):
        #contours_poly[i] = cv2.approxPolyDP(c,100,True)
        #boundRect[i] = cv2.boundingRect(contours_poly[i])
        vx, vy, x, y = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
        line_points[i] = [vx, vy, x, y]
        [pt1, pt2] = converted_line_points(vx, vy, x, y)
        cv2.line (drawing, pt1, pt2, green, 5)

    print([np.arctan2(line_points[0][1], line_points[0][0]),np.arctan2(line_points[1][1], line_points[1][0])])
    average_gradient = np.tan(np.average([np.arctan2(line_points[0][1], line_points[0][0]),np.arctan2(line_points[1][1], line_points[1][0])]))
    average_coords_x = np.average([line_points[0][2], line_points[1][2]])
    average_coords_y = np.average([line_points[0][3], line_points[1][3]])
    [pt1, pt2] = converted_line_points(1, average_gradient, average_coords_x, average_coords_y)
    cv2.line (drawing, pt1, pt2, green, 5)
    
    cv2.drawContours(drawing, line_contours, -1, green, 2, cv2.LINE_AA)
    
    cv2.waitKey(0)

    resized = cv2.resize(drawing, (1228, 921))
    cv2.imshow("Binarised Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    init()
