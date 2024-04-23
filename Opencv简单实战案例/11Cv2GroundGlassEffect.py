"""
简单毛玻璃效果
    1、读取图像：使用OpenCV的cv2.imread()函数读取图像文件。
    2、毛玻璃处理：对图像的每个像素周围的邻域进行随机采样，并将该区域内的像素值随机选取为当前像素值，以模拟毛玻璃效果。
    3、显示图像：使用OpenCV的cv2.imshow()函数显示处理后的图像，或将处理后的图像保存为新的图像文件。
"""


import cv2
import numpy as np
import random


def glass_effect(image, radius=5):
    """
    毛玻璃效果
    :param image:原图
    :param radius:邻域半径
    :return:
    """
    height, width = image.shape[:2]
    result = np.copy(image)
    for y in range(height):
        for x in range(width):
            rand_y = int(random.uniform(y - radius, y + radius))
            rand_x = int(random.uniform(x - radius, x + radius))
            # 边界处理
            rand_y = min(height - 1, max(0, rand_y))
            rand_x = min(width - 1, max(0, rand_x))
            result[y, x] = image[rand_y, rand_x]
    return result


def main():
    # 读取图像
    image = cv2.imread('Images/DogFace.jpg')

    # 应用毛玻璃效果
    blurred_image = glass_effect(image, radius=5)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Dog', image)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("GroundGlassEffect", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('GroundGlassEffect', blurred_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
