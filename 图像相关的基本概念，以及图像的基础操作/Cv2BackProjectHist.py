"""
图像反向投影
    1)首先读入两幅图片，sample对应的图像比较大，target对应的图像比较小，
    2)然后将两幅图像都转换成HSV空间，计算sample图像的直方图roiHist，
    3)接着调研归一化函数normalize()并对其做归一化处理，归一化到0～255区间，
    4)然后输入target图像的hsv空间的target_hsv图像和sample图像归一化了的直方图roiHist，反向计算出图像dst并显示出来
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    sample = cv2.imread("Images/BackProjectHist01.png")
    target = cv2.imread("Images/BackProjectHist02.png")

    # 图像转HSV空间
    roi_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("sample", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("sample", sample)
    # 设置窗口属性，并显示图片
    cv2.namedWindow("target", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("target", target)

    # 计算直方图
    roiHist = cv2.calcHist([roi_hsv], [0, 1], None, [32, 30], [0, 180, 0, 256])
    # 直方图归一化
    cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)

    # 直方图反向投影计算
    dst = cv2.calcBackProject([target_hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Back Projection Demo", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Back Projection Demo", dst)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
