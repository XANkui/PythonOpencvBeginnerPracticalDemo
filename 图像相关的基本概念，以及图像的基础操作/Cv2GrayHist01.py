"""
绘制灰度图像直方图
    1)按照一张输入图像的灰度格式，x坐标代表0～255级的bins，y坐标代表不同bins下的像素个数。
"""

import cv2
from matplotlib import pyplot as plt


def main():
    img = cv2.imread("Images/Gray_Dog.jpg")

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", img)

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # 新建一个图像
    plt.figure()
    # 图坐标图标题
    plt.title("Grayscale Histogram")
    # 图像 x 轴标签
    plt.xlabel("Bins")
    # 图像 y 轴标签
    plt.ylabel("# of Pixels")
    # 画图
    plt.plot(hist)
    # 设置 x 轴 的坐标范围
    plt.xlim([0, 256])
    # 显示图坐标
    plt.show()

    # input()
    cv2.waitKey()
    # 关闭所有的窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
