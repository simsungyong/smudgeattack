from numpy import *
from matplotlib import pyplot as plt
from PIL import Image
import glob
import math
import sys 
import os

def calculateSize(files):
    size_x_ = []
    size_y_ = []
    for file in files:
        image = Image.open(file)
        size_x_.append(image.size[0])
        size_y_.append(image.size[1])
    print(size_x_)
    print(size_y_)

def merge(dir_1, dir_2):
    files_1 = glob.glob(dir_1 + "*.*")
    files_2 = glob.glob(dir_2 + "*.*")

    num_files = len(files_1)
    print(dir_1.split(os.path.sep))


    num1 = (dir_1.split(os.path.sep)[-2]).split("_")[-1]
    num2 = (dir_2.split(os.path.sep)[-2]).split("_")[-1]

    if int(num1) < 10:
        num1 = "0" + num1
    if int(num2) < 10:
        num2 = "0" + num2
  
    for i in range(1, 151):
            image1 = Image.open(dir_1 + "{}.jpg".format(i))
            image2 = Image.open(dir_2 + "{}.jpg".format(i))

            image1 = asarray(image1)
            image2 = asarray(image2)

            result = image1 + image2
            result = Image.fromarray(result)
            
            result.save("../photo_2/2/" + num1 + "-" + num2 + '/{}.jpg'.format(i))

    return result

if __name__ == '__main__':

    for i in range(1, 14):
        for j in range(i+1, 15):
            target_dir_1 = "../photo_rectangle_2/label_{}/".format(i)
            target_dir_2 = "../photo_rectangle_2/label_{}/".format(j)
            size_x = 78
            size_y = 78
    #result = Image.new("L",(size_x, size_y))

            result = merge(target_dir_1, target_dir_2)

    