"""
彩色图像直方
    1)R、G、B三种颜色都绘制在同一张直方图上，可以看出相同bins下不同颜色的数目分布
"""

import cv2
from matplotlib import pyplot as plt


def main():
    img = cv2.imread("Images/Dog.jpg")

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", img)

    chans = cv2.split(img)
    colors = ('b', 'g', 'r')

    # 新建一个图像
    plt.figure()
    # 图像的标题
    plt.title(" Flattened Color Hisogram")
    # x 轴标签
    plt.xlabel("Bins")
    # y 轴标签
    plt.xlabel("# of Pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    # 显示图像
    plt.show()

    # input()
    cv2.waitKey()
    # 关闭所有的窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
