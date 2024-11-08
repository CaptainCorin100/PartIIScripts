import cv2
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import imutils

dropdown_magnifications = ["50X", "10X"]
selected_mag = None

kernel_size = (15,15)
kernel_elem = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

threshold_cutoff = 170

sampling_rate = 2

green = (0,255,0)

def init():
    global selected_mag
    root = tk.Tk()
    root.title("Part II Scripts")
    root.geometry("350x300")

    selected_mag = tk.StringVar(root)
    selected_mag.set("50X")
    mag_dropdown = tk.OptionMenu(root, selected_mag, *dropdown_magnifications)
    mag_dropdown.pack(padx=20, pady=20)

    line_button = tk.Button(root,text="Analyse Line Thickness", command=analyse_line_thickness, anchor="center")
    line_button.pack(padx=20,pady=20)

    void_button = tk.Button(root,text="Analyse Void Fraction", command=analyse_void_fraction, anchor="center")
    void_button.pack(padx=20,pady=20)
    
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

    #Define function to change between coordinate measuring system
    def converted_line_points (vx, vy, x, y):
        lefty = int((-x*vy/vx) + y)
        righty = int(((img.shape[1]-x)*vy/vx) + y)
        return [(img.shape[1] - 1, righty), (0, lefty)]

    #contours_poly = [None]*len(line_contours)
    #boundRect = [None]*len(contours)
    line_points = [None]*len(contours)

    #Calculate equations of lines made by average of contours
    for i,c in enumerate(line_contours):
        #contours_poly[i] = cv2.approxPolyDP(c,100,True)
        #boundRect[i] = cv2.boundingRect(contours_poly[i])
        vx, vy, x, y = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
        line_points[i] = [vx, vy, x, y]
        [pt1, pt2] = converted_line_points(vx, vy, x, y)
        cv2.line (drawing, pt1, pt2, green, 5)

    #Calculate equation of average line between two other lines
    average_gradient = np.tan(np.average([np.arctan2(line_points[0][1], line_points[0][0]),np.arctan2(line_points[1][1], line_points[1][0])]))
    reciprocal_gradient = -1 * 1/average_gradient
    average_coords_x = np.average([line_points[0][2], line_points[1][2]])
    average_coords_y = np.average([line_points[0][3], line_points[1][3]])
    [pt1, pt2] = converted_line_points(1, average_gradient, average_coords_x, average_coords_y)
    cv2.line (drawing, pt1, pt2, green, 5)

    #Calculate shortest distance to other contour from each point on contour
    distance_bin_0 = [None]*len(line_contours[0])
    distance_bin_1 = [None]*len(line_contours[1])
    for i,c in enumerate(line_contours[0]):
        distance_bin_0[i] = convert_pixel_to_micron(cv2.pointPolygonTest(line_contours[1], (int(c[0][0]), int(c[0][1])), True))
    for i,c in enumerate(line_contours[1]):
        distance_bin_1[i] = convert_pixel_to_micron(cv2.pointPolygonTest(line_contours[0], (int(c[0][0]), int(c[0][1])), True))

    #Plot contours on drawing
    cv2.drawContours(drawing, line_contours, -1, green, 2, cv2.LINE_AA)
    cv2.waitKey(0)

    #Resize and draw image
    resized = imutils.resize(drawing, width=1228)     #cv2.resize(drawing, (1228, 921))
    rotated = imutils.rotate(resized, angle=np.rad2deg(np.arctan(average_gradient)))
    cv2.imshow("Binarised Image", rotated)
    
    #Produce histogram of calculated finger widths at points
    combined_distances = distance_bin_0 + distance_bin_1
    print("Finger width has mean of {} um and standard deviation of {} um".format(np.mean(combined_distances), np.std(combined_distances)))
    plt.hist(combined_distances, bins=50, density=True)
    plt.xlabel("Thickness (um)")
    plt.ylabel("Frequency")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.suptitle("Line Thicknesses")
    plt.axvline(np.mean(combined_distances), color="k")
    plt.show()
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def analyse_void_fraction():
    file_path = filedialog.askopenfilename(filetypes=[("JPEG images", "*.jpg")])
    print(file_path)

    #Turn image into black and white
    img = cv2.imread(file_path)
    cv2.waitKey(0)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.blur(grey_img, (3,3))
    th, threshold_img = cv2.threshold(blurred_img, threshold_cutoff, 255, cv2.THRESH_OTSU)

    #Generate blank dataset for black fraction
    point_count = [None] * threshold_img.shape[0]

    for y in range(threshold_img.shape[0]):
        counter = 0
        for x in range(threshold_img.shape[1]):
            if threshold_img[y,x] == 0:
                counter += 1
        point_count[y] = counter * 100 / threshold_img.shape[1]

    print("Total black pixel fraction = {} %".format( (1 - (cv2.countNonZero(threshold_img) / (threshold_img.shape[0] * threshold_img.shape[1])))*100 ))

    params = cv2.SimpleBlobDetector().Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 100
    params.filterByCircularity = True
    params.minCircularity = 0.54
    params.maxCircularity = 1
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector().create(params)
    inverted_img = np.invert(grey_img)
    keypoints = detector.detect(inverted_img)

    keypointed_img = cv2.drawKeypoints(inverted_img, keypoints, np.array([]), green, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.waitKey(0)

    resized = imutils.resize(keypointed_img, width=1228) #cv2.resize(keypointed_img, (1228, 921))
    cv2.imshow("Binarised Image", resized)

    point_indices = range(len(point_count), 0, -1)

    plt.scatter(point_count, point_indices)
    plt.imshow(imutils.opencv2matplotlib(resized))
    plt.ylabel("Image Position")
    plt.xlabel("Void Fraction (%)")
    plt.xlim(0,100)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.show()

def convert_pixel_to_micron(value):
    #Currently hardcoded for 50X image, based on what Affinity reports this to be (100 um = 1975.3 px). Probably not accurate
    if selected_mag.get() == "50X":
        return abs((value*100/1975.3))
    elif selected_mag.get() == "10X":
        return abs((value*100/386.4))

if __name__ == "__main__":
    init()
