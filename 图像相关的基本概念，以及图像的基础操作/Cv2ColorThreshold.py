"""
彩色图像二值化
    1)高于127的像素全部置为255，低于的全部置为0
"""

import cv2


def main():
    # 读取图片，并且灰度处理
    img = cv2.imread("Images/save_Dog.jpg", 0)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", img)

    # 图像二值化
    thresh, dst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("dst", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("dst", dst)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
