"""
简单图片添加水印效果
"""

import cv2


def add_watermark_text(image, text, position='bottom-right', x=None, y=None, font=cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale=1.0, font_color=(255, 255, 255), thickness=1):
    """
    简单添加文字水印效果
    :param image:
    :param text:
    :param position:
    :param x:
    :param y:
    :param font:
    :param font_scale:
    :param font_color:
    :param thickness:
    :return:
    """
    # 复制原始图像，以免修改原始图像
    result = image.copy()

    # 确定水印文本的位置
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    image_height, image_width = image.shape[:2]
    if position == 'top-left':
        text_position = (10, text_size[1] + 10)
    elif position == 'top-right':
        text_position = (image_width - text_size[0] - 10, text_size[1] + 10)
    elif position == 'bottom-left':
        text_position = (10, image_height - 10)
    elif position == 'center':
        text_position = ((image_width - text_size[0]) // 2, (image_height + text_size[1]) // 2)
    else:  # 默认为 'bottom-right'
        text_position = (image_width - text_size[0] - 10, image_height - 10)

    # 如果用户指定了位置，则使用用户指定的位置
    if x is not None and y is not None:
        text_position = (x, y)

    # 添加水印文本
    cv2.putText(result, text, text_position, font, font_scale, font_color, thickness)

    return result


def add_watermark_image(image, watermark_image, position='bottom-right', x=None, y=None):
    """
    简单添加图片水印效果
    :param image:
    :param watermark_image:
    :param position:
    :param x:
    :param y:
    :return:
    """
    # 复制原始图像，以免修改原始图像
    result = image.copy()

    # 确定水印图片的位置
    watermark_height, watermark_width = watermark_image.shape[:2]
    image_height, image_width = image.shape[:2]
    if position == 'top-left':
        watermark_position = (0, 0)
    elif position == 'top-right':
        watermark_position = (image_width - watermark_width, 0)
    elif position == 'bottom-left':
        watermark_position = (0, image_height - watermark_height)
    elif position == 'center':
        watermark_position = ((image_width - watermark_width) // 2, (image_height - watermark_height) // 2)
    else:  # 默认为 'bottom-right'
        watermark_position = (image_width - watermark_width, image_height - watermark_height)

    # 如果用户指定了位置，则使用用户指定的位置
    if x is not None and y is not None:
        watermark_position = (x, y)

    # 获取水印图片的 alpha 通道
    watermark_alpha = watermark_image[:, :, 3] / 255.0

    # 提取水印图片的 BGR 通道
    watermark_bgr = watermark_image[:, :, :3]

    # 将水印图片叠加到原始图像上
    for c in range(3):
        result[watermark_position[1]:watermark_position[1] + watermark_height,
        watermark_position[0]:watermark_position[0] + watermark_width, c] = \
            (1 - watermark_alpha) * result[watermark_position[1]:watermark_position[1] + watermark_height,
                                    watermark_position[0]:watermark_position[0] + watermark_width, c] + \
            watermark_alpha * watermark_bgr[:, :, c]

    return result


def main():
    # 调用函数并指定输入图像、水印和输出图像文件路径
    input_image_path = "Images/DogFace.jpg"
    watermark_path = "Images/Watermark.png"
    output_image_path = "Images/DogFace_Watermark.jpg"

    image = cv2.imread(input_image_path)

    # 设置窗口属性，并显示图片
    cv2.namedWindow("Dog", cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Dog', image)

    output_image = add_watermark_text(image, "Water mark", x=200, y=300, font_scale=2.0, thickness=2)

    # 保存处理后的图像
    # 设置窗口属性，并显示图片
    cv2.namedWindow("add_watermark_text", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("add_watermark_text", output_image)

    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    output_image = add_watermark_image(image, watermark, position="center")

    # 保存处理后的图像
    # 设置窗口属性，并显示图片
    cv2.namedWindow("add_watermark_image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("add_watermark_image", output_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
