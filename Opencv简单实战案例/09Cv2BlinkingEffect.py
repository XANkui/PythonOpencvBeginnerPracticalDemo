"""
简单的闪烁效果
    1、读取图像文件： 首先，从文件系统中读取输入的图像文件，该图像将作为闪烁效果的基础。

    2、设定闪烁周期： 确定闪烁的周期，即图像亮度和对比度调整的时间间隔。在本例中，设定为1秒。

    3、进入处理循环： 在一个无限循环中，持续处理图像以实现闪烁效果。

    4、计算时间差： 在每次循环迭代中，计算当前时间与开始闪烁的时间之间的时间差。

    5、调整亮度和对比度： 如果时间差大于闪烁周期，则进行图像亮度和对比度的调整。调整值通常在一定范围内随机生成，以获得随机的闪烁效果。

    6、应用调整后的效果： 使用 cv2.convertScaleAbs() 函数将调整后的亮度和对比度应用于原始图像，生成调整后的图像。

    7、反转亮度： 如果时间差大于闪烁周期，则将调整后的图像的亮度反转，以模拟闪烁效果。

    8、显示处理后的图像： 使用 cv2.imshow() 函数在窗口中显示处理后的图像。

    9、等待用户退出： 检测用户是否按下 'q' 键，如果是则退出循环。

    10、释放资源： 循环结束后，释放窗口资源并结束程序。
"""

import cv2
import numpy as np
import time


def BlinkingEffect(image, blink_interval=1):
    """
    简单闪烁效果
    :param image:
    :param blink_interval: 闪烁间隔时间
    :return:
    """

    # 检查图像是否成功读取
    if image is None:
        print("Error: Unable to read image.")
        exit()

    # 定义闪烁周期（秒）
    blink_interval = blink_interval

    # 定义开始闪烁的时间
    start_blink_time = time.time()

    # 循环处理图像
    while True:
        # 计算当前时间和开始闪烁的时间之间的时间差
        current_time = time.time()
        time_diff = current_time - start_blink_time

        # 计算亮度和对比度的调整值
        brightness = np.random.uniform(-50, 50)
        contrast = np.random.uniform(0.5, 1.5)

        # 使用亮度和对比度调整值调整图像
        adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

        # 如果时间差大于闪烁周期，则进行图像闪烁处理
        if time_diff > blink_interval:
            # 反转图像亮度
            adjusted_image = 255 - adjusted_image

            # 更新开始闪烁的时间
            start_blink_time = current_time

        # 设置窗口属性，并显示图片
        cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Neon Light', adjusted_image)

        # 按下 q 键，退出
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 释放窗口
    cv2.destroyAllWindows()


def main():

    # 读取图像文件
    image = cv2.imread('Images/DogFace.jpg')

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", image)

    # 调用闪烁效果
    BlinkingEffect(image, 1)


if __name__ == "__main__":
    main()
