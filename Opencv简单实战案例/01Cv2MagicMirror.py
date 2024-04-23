"""
哈哈镜效果
"""

import cv2
import math


def EnlargeEffectMirror(img, radius):
    """
    哈哈镜放大效果
    :param img: 图片
    :param radius: 半径
    :return: 变化后的数据
    """
    # 获取图片的高、宽、和维度
    h, w, n = img.shape
    cx = w / 2
    cy = h / 2

    r = int(radius / 2.0)
    new_img = img.copy()

    # 遍历调整图片
    for i in range(w):
        for j in range(h):
            tx = i - cx
            ty = j - cy
            distance = tx * tx + ty * ty
            if distance < radius * radius:
                x = int(int(tx / 2.0) * (math.sqrt(distance) / r) + cx)
                y = int(int(ty / 2.0) * (math.sqrt(distance) / r) + cy)
                if x < w and y < h:
                    new_img[j, i, 0] = img[y, x, 0]
                    new_img[j, i, 1] = img[y, x, 1]
                    new_img[j, i, 2] = img[y, x, 2]
    return new_img


def ReduceEffectMirror(img, compress):
    """
    哈哈镜缩小效果
    :param img:
    :param compress: 图像缩小数值，越大，压缩越严重
    :return:
    """
    height, width, n = img.shape
    center_x = width / 2
    center_y = height / 2
    new_data = img.copy()
    # 图像遍历
    for i in range(width):
        for j in range(height):
            tx = i - center_x
            ty = j - center_y
            theta = math.atan2(ty, tx)
            radius = math.sqrt((tx * tx) + (ty * ty))
            newx = int(center_x + (math.sqrt(radius) * compress * math.cos(theta)))
            newy = int(center_y + (math.sqrt(radius) * compress * math.sin(theta)))
            # 防止计算后坐标小于0
            if newx < 0 and newx > width:
                newx = 0
            if newy < 0 and newy > height:
                newy = 0
            if newx < width and newy < height:
                new_data[j, i][0] = img[newy, newx][0]
                new_data[j, i][1] = img[newy, newx][1]
                new_data[j, i][2] = img[newy, newx][2]

    return new_data


def TestEnlargeEffectMirror():
    """
    测试哈哈镜放大效果
    :return: null
    """

    img = cv2.imread("Images/DogFace.jpg")

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", img)

    # 该值可以自行定义，它决定了哈哈镜的大小，当图像很大时，应该相应的调大
    enlarge_img = EnlargeEffectMirror(img, 400)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("enlarge_img", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("enlarge_img", enlarge_img)


def TestReduceEffectMirror():
    """
    测试哈哈镜缩小效果
    :return: null
    """

    img = cv2.imread("Images/DogFace.jpg")

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", img)

    # 该值可以自行定义，它决定了哈哈镜的大小，当图像很大时，应该相应的调大
    reduce_img = ReduceEffectMirror(img, 12)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("reduce_img", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("reduce_img", reduce_img)


def main():
    # TestEnlargeEffectMirror()

    TestReduceEffectMirror()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
