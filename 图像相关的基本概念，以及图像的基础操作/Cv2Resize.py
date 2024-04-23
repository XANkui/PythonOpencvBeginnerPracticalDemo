"""
图像的缩放
    1)输入的原图，
    2)调用了resize()函数，
    3)一张图片是整体尺寸缩小到了100×100，
    4）一张图片是等比例放大2倍。
    5）resize()函数既可以指定输出图像的具体尺寸，也可以指定图像水平或垂直缩放的比例。
"""

import cv2


def main():
    img = cv2.imread("Images/Dog.jpg")

    # 图片尺寸
    height, width, n = img.shape

    # 缩小
    downscale = cv2.resize(img, (100, 100), interpolation=cv2.INTER_LINEAR)
    # 放大
    upscale = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_LINEAR)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("downscale", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("downscale", downscale)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("upscale", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("upscale", upscale)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", img)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()
