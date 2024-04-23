"""
灰度图像直方图均衡化(全局)
"""

import cv2
import numpy as np


def main():
    # 读取图片并且灰度化
    img = cv2.imread("Images/Cullet.jpg", 0)

    # 图像均衡化
    eq = cv2.equalizeHist(img)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Histogram Equalization", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Histogram Equalization", np.hstack([img, eq]))

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
