import csv
import matplotlib.pyplot as plt
from tkinter import filedialog
import numpy as np


def main():
    for data_name in ["Under-sintered", "Over-sintered"]:
        print("staet")
        file_path = filedialog.askopenfilename(filetypes=[("{} CSV".format(data_name), "*.csv")])
        depths = []
        counts = []

        with open(file_path, "r") as csvfile:
            lines = csv.reader(csvfile, delimiter=",")
            next(lines)
            for row in lines:
                depths.append(float(row[0]))
                counts.append(float(row[1]))

        z = np.polyfit(x=depths,y=counts,deg=1)
        p = np.poly1d(z)

        plt.plot(depths,counts)
        plt.plot(depths, p(depths))
        
        plt.pause(0.1)
        print("yep")


    plt.ylabel("Counts", fontsize=20)
    plt.xlabel("Height (μm)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.legend(["Under-sintered", "Undersintered Fit", "Over-sintered", "Oversintered Fit"])
    plt.show()


if __name__ == "__main__":
    main()
