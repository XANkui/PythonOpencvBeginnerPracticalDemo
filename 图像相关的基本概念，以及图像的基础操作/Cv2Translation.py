"""
图像平移
    1）首先构建变换矩阵M，设置向左、向下各平移50个像素，则M=[[1,0,-100],[0,1,100]。
    2）图片的平移并没有设置改变图像的尺寸，因此平移后无像素的地方显示为黑色。
"""

import numpy as np
import cv2


def main():
    img = cv2.imread('Images/Dog.jpg')
    height, width, n = img.shape
    M = np.array([[1, 0, 100], [0, 1, 100]], np.float32)

    # 图像平移
    img_tr = cv2.warpAffine(img, M, img.shape[:2])

    # 设置窗口属性，并显示图片
    cv2.namedWindow("img_tr", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('img_tr', img_tr)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Dog', img)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
