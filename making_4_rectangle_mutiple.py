#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

def img_trim(img, x, y, w, h):
    
    img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
    w1,h1 = img_trim.shape[0:2]
    
    return img_trim


def canny(image):
    # Apply Canny Edge detection. 
    edges = cv.Canny(image, 60, 20)
    #w,h = edges.shape[::-1]
    #print(w, h)
    return edges


def is_grey_scale(img):
    img = img.convert('RGB')

    w,h = img.size
    for i in range(w):
        for j in range(h):
            r,g,b = img.getpixel((i,j))
            if r != g != b: return False
    return True

if __name__ == "__main__":
    
    for i in range(1, 31):
        
        img = cv.imread('../photo/scenario/8/sns/{}.jpg'.format(i))

        img = cv.resize(img, dsize=(600, 900), interpolation=cv.INTER_AREA)
        #template = cv.resize(template, dsize=(600, 900), interpolation=cv.INTER_AREA)

        kernel = np.ones((12,12), np.uint8)
        kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(3,3))
        img_noise = cv.morphologyEx(img, cv.MORPH_ELLIPSE, kernel1)
        ret, thresh = cv.threshold(img_noise,230,255, cv.THRESH_BINARY)

        refer11 = canny(thresh)
        result = cv.dilate(refer11, kernel, iterations = 1)
        contours, hierachy = cv.findContours(result, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

        contour_list = []

        for cnt in contours:
            m = len(cnt)
            contour_list.append(m)
        max_ = max(contour_list)
        
        index_ = contour_list.index(max_)
        contour_list.remove(max_)
        screen = max(contour_list)
        contour_list.insert(index_, max_)
        
        cnt = contours[contour_list.index(screen)]
        epsilon = 0.1 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        

        x_list = []
        y_list = []
        
        for appro in approx:
            for coord in appro:
                x_list.append(coord[0])
        for appro in approx:
            for coord in appro:
                y_list.append(coord[1])    

        first_min_x = min(x_list)
        x_list.remove(first_min_x)
        standard_x_1 = min(x_list)
        x_list.remove(standard_x_1)
        standard_x_2 = min(x_list)
        
        first_min_y = min(y_list)
        y_list.remove(first_min_y)
        standard_y_1 = min(y_list)
        y_list.remove(standard_y_1)
        standard_y_2 = min(y_list)
        
        img_t = img_trim(img, standard_x_1, standard_y_1,standard_x_2-standard_x_1, standard_y_2-standard_y_1)
        cv.drawContours(img, [approx] , -1, (0,255,0), 10)

        # Trim 3x3 
        img_t1 = img_trim(img_t, 10, 140,130, 130)
        t_noise = cv.morphologyEx(img_t1, cv.MORPH_ERODE, kernel, iterations=1)

        width, height = img_t1.shape[:2]
        print(width, height)
        ttt1 = canny(img_t1)

        # Trim 2x2
        img_t1_1 = img_trim(t_noise, 0, 3, 78, 78)
        img_t1_2 = img_trim(t_noise, 52,0, 78, 78)
        img_t1_3 = img_trim(t_noise, 0, 55, 78, 78)
        img_t1_4 = img_trim(t_noise, 52, 55, 78, 78)

        ttt_1 = canny(img_t1_1)
        ttt_2 = canny(img_t1_2)
        ttt_3 = canny(img_t1_3)
        ttt_4 = canny(img_t1_4)

        cv.imwrite('../photo_rectangle_2/scenario/8/sns/{}_1.jpg'.format(i), ttt_1)
        cv.imwrite('../photo_rectangle_2/scenario/8/sns/{}_2.jpg'.format(i), ttt_2)
        cv.imwrite('../photo_rectangle_2/scenario/8/sns/{}_3.jpg'.format(i), ttt_3)
        cv.imwrite('../photo_rectangle_2/scenario/8/sns/{}_4.jpg'.format(i), ttt_4)

    #cv.imshow('result',img_t1_4) # 결과 이미지 출력

    #Type convert from numarry to Image.
    print(img_t1_1.shape[:2])
    print(type(ttt_1))
    type_convert = Image.fromarray(ttt_1)
    print(type(type_convert))
    #Checking gray scale
    result = is_grey_scale(type_convert)
    print(result)

    cv.waitKey(0)