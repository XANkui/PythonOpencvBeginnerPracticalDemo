"""
图像镜像变换
"""

import cv2


def main():
    img = cv2.imread("Images/Dog.jpg")

    # 水平翻转
    xImg = cv2.flip(img, 1, dst=None)

    # 垂直翻转
    yImg = cv2.flip(img, 0, dst=None)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("xImg Horizontal", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("xImg Horizontal", xImg)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("yImg Vertical", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("yImg Vertical", yImg)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog Origin", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog Origin", img)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
