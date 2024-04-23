"""
彩色图像和灰度图像的转换
    1)主要实现原理：增加图像通道
    2）新建了一个3通道的空的彩色图像，
    3）然后将读取的灰度图像放在新建的彩色图像的第一个通道，也就是B通道，
    4）其他两个通道赋值0，所以图像整体呈现蓝色
"""

import numpy as np
import cv2


def main():
    img = cv2.imread("Images/Dog.jpg")

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", img)

    # 获取图片长宽
    height, width, n = img.shape

    # 生成一个空的彩色图像
    gray = np.zeros((height, width, 3), np.uint8)

    # 遍历像素，0 通道赋值，其余通道 0
    for i in range(height):
        for j in range(width):
            gray[i, j][0] = img[i, j][0]
            gray[i, j][1] = 0
            gray[i, j][2] = 0

    # 设置窗口属性，并显示图片
    cv2.namedWindow("grayToColor", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("grayToColor", gray)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
