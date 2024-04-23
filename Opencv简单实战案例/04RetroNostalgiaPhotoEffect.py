"""
复古怀旧风格
    怀旧风格的设计主要是在图像的颜色空间进行处理，
    以GRB空间为例，对R、G、B这3个通道的颜色数值进行处理，
    让图像有一种泛黄的老照片效果。设计的转换公式如下：
        B = int(0.272 * r + 0.534 * g + 0.131 * b)
        G = int(0.349 * r + 0.686 * g + 0.168 * b)
        R = int(0.393 * r + 0.769 * g + 0.189 * b)
"""

import cv2
import numpy as np


def RetroEffect(img):
    """
    复古怀旧风格
    :param img:
    :return:
    """
    copy_img = img.copy()

    height, width, n = img.shape

    # 遍历像素处理
    for i in range(height):
        for j in range(width):
            b = img[i, j][0]
            g = img[i, j][1]
            r = img[i, j][2]

            # 计算新的图像中的 RGB 值
            B = int(0.272 * r + 0.534 * g + 0.131 * b)
            G = int(0.349 * r + 0.686 * g + 0.168 * b)
            R = int(0.393 * r + 0.769 * g + 0.189 * b)

            # 约束图像像素值，防止溢出
            copy_img[i, j][0] = max(0, min(B, 255))
            copy_img[i, j][1] = max(0, min(G, 255))
            copy_img[i, j][2] = max(0, min(R, 255))

        # 添加颗粒效果
        # noise = np.random.normal(0, 0.3, img.shape).astype(np.uint8)
        # copy_img = cv2.add(copy_img, noise)

    return copy_img


def main():
    img = cv2.imread("Images/DogFace.jpg")
    retro_img = RetroEffect(img)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", img)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("retro_img", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("retro_img", retro_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
