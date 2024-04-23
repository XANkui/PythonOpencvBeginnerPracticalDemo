"""
素描画风格效果

    （1）去色，将彩色图片转换成灰度图像。

    图像的打开可以通过cv2.imread代码打开，cv2.cvtColor可以将图片转化为灰度图。你也可以在读取图片的时候增加一个额外的参数使得图像直接转化为灰度图

    （2）复制去色图层，并且反色，反色为Y(i,j)=255-X(i,j)。

    灰度图反色图像可以通过将灰度图每个像素点取反得到，由于灰度图的像素点的在0-255之间，将其取反的话就是255-当前像素点。

    （3）对反色图像进行高斯模糊。

    Gaussian blur能够很有效地减少图像中的噪声，能够将图像变得更加平滑一点，在数学上等价于用高斯核来对图像进行卷积操作。我们可以通过cv2.GaussianBlur来实现高斯模糊操作，参数ksize表示高斯核的大小。sigmaX和sigmaY分别表示高斯核在 X 和 Y 方向上的标准差。

    （4）模糊后的图像叠加模式选择颜色减淡效果。

    这一步骤自然就是需要得到最终的素描图结果了。在传统照相技术中，当需要对图片某个区域变得更亮或者变暗，可以通过控制它的曝光时间，这里就用到亮化(Dodging)和暗化(burning)的技术。

"""

import cv2
import numpy as np
 

def dodgeNaive(image, mask):
    """
    该版本，比较耗时，请使用 dogeV2
    :param image:
    :param mask:
    :return:
    """
    # determine the shape of the input image
    width, height = image.shape[:2]
 
    # prepare output argument with same size as image
    blend = np.zeros((width, height), np.uint8)
 
    for col in range(width):
        for row in range(height):
            # do for every pixel
            if mask[col, row] == 255:
                # avoid division by zero
                blend[col, row] = 255
            else:
                # shift image pixel value by 8 bits
                # divide by the inverse of the mask
                tmp = (image[col, row] << 8) / (255 - mask)
                # print('tmp={}'.format(tmp.shape))
                # make sure resulting value stays within bounds
                if tmp.any() > 255:
                    tmp = 255
                    blend[col, row] = tmp
 
    return blend
 
 
def dodgeV2(image, mask, scale):
    """
    灰度图与高斯模糊底片的融合
    :param image:
    :param mask:
    :param scale: 风格化效果，值显示的效果不同，越大，越白
    :return:
    """
    return cv2.divide(image, 255 - mask, scale=scale)
 
 
def burnV2(image, mask, scale):
    """
    灰度图与高斯模糊底片的融合
    :param image:
    :param mask:
    :param scale:风格化效果，值显示的效果不同，越大，越暗
    :return:
    """
    return 255 - cv2.divide(255 - image, 255 - mask, scale=scale)
 
 
def rgb_to_sketch(src_image_name, dst_image_name):
    """

    :param src_image_name: 原始图片
    :param dst_image_name: 要保留的风格化图片
    :return:
    """
    img_rgb = cv2.imread(src_image_name)

    # 将图像转化为灰度图
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # 灰度反色操作
    img_gray_inv = 255 - img_gray

    # 高斯模糊
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                                sigmaX=0, sigmaY=0)

    # 素描画风格处理
    img_blend = dodgeV2(img_gray, img_blur, 250)
    # img_blend = burnV2(img_gray, img_blur, 250)

    # 图片显示
    # 设置窗口属性，并显示图片
    cv2.namedWindow("original", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('original', img_rgb)
    cv2.namedWindow("gray", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('gray', img_gray)
    cv2.namedWindow("gray_inv", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('gray_inv', img_gray_inv)
    cv2.namedWindow("gray_blur", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('gray_blur', img_blur)
    cv2.namedWindow("sketch drawing effect", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("sketch drawing effect", img_blend)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存图片
    cv2.imwrite(dst_image_name, img_blend)
 
 
if __name__ == '__main__':
    src_image_name = 'Images/DogFace.jpg'
    dst_image_name = 'Images/sketch_example.jpg'
    rgb_to_sketch(src_image_name, dst_image_name)