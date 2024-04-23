"""
卡通漫画风格效果
    1）载入图像
    2）创建边缘蒙版
    3）减少调色板
    4）将边缘蒙版与经过颜色处理的图像结合起来

"""

import cv2
import numpy as np


def read_file(filename):
    """
    读取显示，并返回图片
    :param filename:
    :return:
    """
    img = cv2.imread(filename)
    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", img)
    return img


def edge_mask(img, line_size, blur_value):
    """
    创建边缘蒙版
    :param img:
    :param line_size:
    :param blur_value:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges


def color_quantization(img, k):
    """
    减少图片色调
    :param img:
    :param k: 调整k值来确定想要应用到图像的颜色数量
    :return:
    """
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


def color_blur(img, d, sigmaColor, sigmaSpace):
    """
    图像简单模糊处理
    :param img:
    :param d:每个像素邻域的直径
    :param sigmaColor:该参数的值越大，表示半等色的区域越大
    :param sigmaSpace:该参数的值越大，意味着较远的像素只要其颜色足够接近，就会相互影响
    :return:
    """
    blurred = cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return blurred


def img_and(blurred, edges):
    """
    将边缘蒙版与经过颜色处理的图像相结合
    :param blurred:
    :param edges:
    :return:
    """
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    return cartoon


def main():
    img = read_file("Images/DogFace.jpg")
    edge_img = edge_mask(img, 11, 7)
    color_img = color_quantization(img, 9)
    blurred_color_img = color_blur(color_img, 7, 7, 200)
    cartoon_effect = img_and(blurred_color_img, edge_img)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("cartoon_effect", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("cartoon_effect", cartoon_effect)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
