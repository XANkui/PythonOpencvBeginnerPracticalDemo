"""
简单颜色反转效果
    1、灰度反转： 将彩色图像转换为灰度图像，然后将每个像素的灰度值取反。
    2、彩色反转： 将每个通道的像素值取反，可以通过255减去原始像素值来实现。
"""


import cv2


# 灰度反转
def grayscale_invert(image):
    """
    灰度反转
    :param image:
    :return:
    """
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 灰度反转
    inverted_image = 255 - gray_image
    return inverted_image


# 彩色反转
def color_invert(image):
    """
    彩色反转
    :param image:
    :return:
    """
    # 彩色反转
    inverted_image = 255 - image
    return inverted_image


def main():
    # 读取图像
    image = cv2.imread('Images/DogFace.jpg')

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Dog', image)

    # 灰度反转
    inverted_gray = grayscale_invert(image)
    # 彩色反转
    inverted_color = color_invert(image)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Grayscale Inverted", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Grayscale Inverted', inverted_gray)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Color Inverted", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Color Inverted', inverted_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
