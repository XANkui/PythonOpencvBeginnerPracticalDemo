"""
读取一张图片并显示和存储
（1）在代码中加入一行import cv2，就完成了OpenCV的包导入。

（2）调用函数的时候需要在OpenCV原本的函数前加上cv2.，以确保能找到该函数。

（3）注意Python的缩进方式，它代表了函数的范围。

（4）imwrite()函数可以将图像保存成不同的格式。
"""

import cv2


def main():
    img = cv2.imread("Images/Dog.jpg")
    cv2.imshow("Dog", img)
    cv2.imwrite("Images/save_Dog.jpg", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
