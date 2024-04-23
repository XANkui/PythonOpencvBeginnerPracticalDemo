
"""
读取一张图片保存为灰度图

（1）在代码中加入一行import cv2，就完成了OpenCV的包导入。

（2）调用函数的时候需要在OpenCV原本的函数前加上cv2.，以确保能找到该函数。

（3）注意Python的缩进方式，它代表了函数的范围。

（4）imwrite()函数可以将图像保存成不同的格式。
"""

import cv2


def main():
    # 读取图片
    image = cv2.imread('Images/Dog.jpg')

    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 保存灰度图
    cv2.imwrite('Images/Gray_Dog.jpg', gray_image)


if __name__ == '__main__':
    main()
