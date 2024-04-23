"""
彩色图像和灰度图像的转换
    1)用预定义好的Colormap（色度图）来给图片上色
    2）伪彩色图像目前主要应用在对高度、压力、密度、湿度等描述上，彩色数据可视化。
"""

import numpy as np
import cv2


def main():
    img = cv2.imread('Images/Gray_Dog.jpg')

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", img)

    # 色度图上色
    img_Color = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("img_Color", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("img_Color", img_Color)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
