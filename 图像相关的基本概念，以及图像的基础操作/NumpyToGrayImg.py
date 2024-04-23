"""
使用 Numpy 生成一幅灰度图
    1）新建个img的二维数组，即10×10的全 125 数组，
    2）然后将数组保存成图片。
    3）运行后，程序目录下多出了一个灰色的图片，边角显示图片的大小为10×10，JPEG格式
"""

import numpy as np
import cv2


def main():
    # 创建一个 4x4 的黑色图片
    img = np.array([
        [125, 125, 125, 125, 125, 125, 125, 125, 125, 125],
        [125, 125, 125, 125, 125, 125, 125, 125, 125, 125],
        [125, 125, 125, 125, 125, 125, 125, 125, 125, 125],
        [125, 125, 125, 125, 125, 125, 125, 125, 125, 125],
        [125, 125, 125, 125, 125, 125, 125, 125, 125, 125],

        [125, 125, 125, 125, 125, 125, 125, 125, 125, 125],
        [125, 125, 125, 125, 125, 125, 125, 125, 125, 125],
        [125, 125, 125, 125, 125, 125, 125, 125, 125, 125],
        [125, 125, 125, 125, 125, 125, 125, 125, 125, 125],
        [125, 125, 125, 125, 125, 125, 125, 125, 125, 125],
    ], dtype=np.uint8)

    # 保存图片，然后显示图片
    cv2.imwrite("Images/Numpy_GrayImg.jpg", img)
    cv2.imshow("Numpy_GrayImg.jpg", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
