"""
灰度图像直方图局部均衡化
"""

import cv2
import numpy as np


def main():
    img = cv2.imread("Images/Cullet.jpg", 0)

    # 创建一个 8x8 的 clash
    clahe = cv2.createCLAHE(5, (8, 8))
    # 图片切分成 8x8 的小块
    dst = clahe.apply(img)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Local Histogram Equalization", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Local Histogram Equalization", np.hstack([img, dst]))

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
