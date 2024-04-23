"""
简单马赛克效果

"""

import cv2


def MosaicEffect(img, start_x, start_y, width, height, mosaic_size=10):
    """
    马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，
    这样可以模糊细节，但是可以保留大体的轮廓。
    :param img:
    :param start_x:马赛克左顶点
    :param start_y:马赛克右顶点
    :param width:马赛克宽
    :param height:马赛克高
    :param mosaic_size:马赛克每一块的宽
    :return:
    """
    fh, fw, n = img.shape
    print("fh = ")
    print(fh)
    print("fw = ")
    print(fw)
    if (start_y + height > fh) or (start_x + width > fw):
        return
    copy_img = img.copy()

    # 减去 mosaic_size 防止溢出
    for i in range(0, height - mosaic_size, mosaic_size):
        for j in range(0, width - mosaic_size, mosaic_size):
            rect = [j + start_x, i + start_y, mosaic_size, mosaic_size]

            # tolist
            color = img[i + start_y][j + start_x].tolist()
            left_up = (rect[0], rect[1])

            # 减去一个像素
            right_down = (rect[0] + mosaic_size - 1, rect[1] + mosaic_size - 1)
            cv2.rectangle(copy_img, left_up, right_down, color, -1)

    return copy_img


def main():
    img = cv2.imread("Images/DogFace.jpg")
    height, width, n = img.shape

    # 整体马赛克效果
    mosaic_whole_img = MosaicEffect(img, 0, 0, width, height)

    # 注意局部长宽
    mosaic_img = MosaicEffect(img, 100, 100, 600 - 100, 300 - 100)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dog", img)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("mosaic_whole_img", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("mosaic_whole_img", mosaic_whole_img)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("mosaic_img", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("mosaic_img", mosaic_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
