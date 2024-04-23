"""
简单图像浮雕效果
    （1）将彩色图像转换为灰度图像。
    （2）对灰度图像进行卷积操作，使用卷积核进行滤波，得到一组新的像素值。卷积核的大小可以根据需要进行调整，通常采用3x3或5x5的大小。
    （3）对于每个像素，将卷积操作后得到的像素值减去该像素在原始图像中的像素值，得到浮雕值。
    （4）根据浮雕值，将像素点的灰度值进行调整，使得局部区域的像素值相对于周围的像素值有一定的增加或减少。
    （5）将处理后的像素值重新映射到0-255的灰度值范围内，生成浮雕效果的图像。

"""


import cv2
import numpy as np


def emboss_effect(img):
    """
    浮雕效果处理
    :param img:
    :return:
    """
    # 将彩色图像转换为灰度图像
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建一个3x3的浮雕卷积核
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])

    # 对灰度图像进行卷积操作
    embossed_image = cv2.filter2D(gray_image, -1, kernel)

    # 对卷积操作后的像素值进行调整，使得局部区域的像素值相对于周围的像素值有一定的增加或减少
    embossed_image = cv2.addWeighted(gray_image, 0.5, embossed_image, 0.5, 0)

    # 将处理后的像素值重新映射到0-255的灰度值范围内
    embossed_image = cv2.convertScaleAbs(embossed_image)

    return embossed_image


def main():
    # 读取图像
    image = cv2.imread('Images/DogFace.jpg')

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", image)

    # 应用浮雕效果
    embossed_image = emboss_effect(image)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Embossed Image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Embossed Image', embossed_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
