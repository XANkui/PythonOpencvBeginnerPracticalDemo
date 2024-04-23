"""
彩色图像的通道分离和混合
    1)输入一幅彩色图像，通过程序将其分割成R、G、B这3个通道的图像并显示。
    2)在分割前需要先确定图像的颜色通道分布，因此先调用cvtColor()函数固定颜色通道
    3)在图像通道分离后，不同颜色通道的图像显示深浅不一，单通道的图像呈现该颜色通道的灰度信息
    4)把这3个颜色通道混合一下，这样img3又回到了原来输入的彩色图像样式
"""

import numpy as np
import cv2


def main():
    # 读取图片
    img = cv2.imread("Images/Dog.jpg")

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", img)

    # 固定 rgb 通道，分离颜色
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img2)

    cv2.namedWindow("Red", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Red", r)

    cv2.namedWindow("Green", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Green", g)

    cv2.namedWindow("Blue", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Blue", b)

    # 合并颜色通道
    mergeImage = cv2.merge([b, g, r])

    cv2.namedWindow("mergeImage", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("mergeImage", mergeImage)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
