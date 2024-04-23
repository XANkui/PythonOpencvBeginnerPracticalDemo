"""
图像旋转
"""

import cv2


def main():
    img = cv2.imread("Images/Dog.jpg")
    height, width, n = img.shape

    # 中心旋转 90°
    M = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)

    img_rotatin = cv2.warpAffine(img, M, img.shape[:2])

    # 设置窗口属性，并显示图片
    cv2.namedWindow("img_rotatin", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('img_rotatin', img_rotatin)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Dog', img)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
