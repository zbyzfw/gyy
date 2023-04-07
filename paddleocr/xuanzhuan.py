
import cv2
import math
import random

import matplotlib.image as im
import numpy as np
import os

def randomErasing(imgDir):
    rotate = '0'
    img = cv2.imread(imgDir)
    if rotate == '1':
        img = cv2.flip(cv2.transpose(img), 1)
    elif rotate == '2':
        img = cv2.flip(img, -1)
    elif rotate == '3':
        img= cv2.flip(cv2.transpose(cv2.flip(img, -1)), 1)
    roi=cv2.imread('D:/aaa01.jpg')
    _, max_val, _, top_left2 = cv2.minMaxLoc(cv2.matchTemplate
                                             (img, roi, cv2.TM_CCOEFF_NORMED))
    roi = img[top_left2[1]:bottom_right2[1], top_left2[0]:bottom_right2[0]]
    cv2.imshow("3",roi )
    cv2.waitKey(1000)
    return roi


if __name__ == '__main__':
    # imgDir = './0001.jpg'
    # img = cv2.imread(imgDir)
    path_name = r'D:/345'
    for item in os.listdir(path=path_name):
        # print(item[-4:])
        if item[-4:] == ".jpg":
            img_path = os.path.join(path_name, item)
            # print(img_path)
            img = randomErasing(img_path)
            cv2.imwrite('D:/378/' + item, roi)