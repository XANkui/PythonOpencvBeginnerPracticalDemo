"""
简单水彩画效果

"""

import cv2


def watercolor_effect(image):
    """
    水彩画效果
    :param image:
    :return:
    """
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 对灰度图像进行中值模糊处理
    blurred = cv2.medianBlur(gray, 15)

    # 对模糊处理后的图像进行边缘检测
    edges = cv2.Laplacian(blurred, cv2.CV_8U, ksize=5)

    # 对边缘图像进行二值化处理
    _, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

    # 对原始图像进行颜色量化
    quantized = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)

    # 将颜色量化后的图像与边缘掩码进行合并
    watercolor = cv2.bitwise_and(quantized, quantized, mask=mask)

    return watercolor


def main():
    # 读取图像
    image = cv2.imread('Images/DogFace.jpg')

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", image)

    # 应用水彩画效果
    watercolor_image = watercolor_effect(image)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Watercolor Image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Watercolor Image', watercolor_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
