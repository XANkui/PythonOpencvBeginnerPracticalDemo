"""
简单去除图片水印效果
    1、选择水印的ROI（感兴趣区域）
    2、自适应检测水印并生成遮罩
    3、生成水印的遮罩
    4、应用遮罩去除水印
    5、保存处理后的图片
"""

import cv2
import numpy as np


def select_roi_for_mask(image):
    """
    从图像中选择水印的ROI
    :param image: 图像数据
    :return: 水印ROI的坐标和尺寸 (x, y, w, h)，如果未选择ROI则返回 None
    """
    if image is None or len(image.shape) != 3:
        raise ValueError("Input image is invalid or not in BGR format.")

    instructions = "Select ROI and press SPACE or ENTER"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, instructions, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    r = cv2.selectROI("Select ROI", image)
    cv2.destroyAllWindows()

    if r[2] == 0 or r[3] == 0:
        print("ROI not selected. Watermark removal aborted.")
        return None

    return r


def detect_watermark_adaptive(image, roi):
    """
    自适应检测水印并生成遮罩。
    :param image: 图像数据
    :param roi: 水印的ROI坐标和尺寸 (x, y, w, h)。
    :return: 水印的遮罩图像数据，如果ROI未选择则返回 None
    """
    if roi is None:
        print("ROI not selected. Watermark removal aborted.")
        return None

    roi_image = image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    gray_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = binary_image

    return mask


def generate_watermark_mask(image, roi):
    """
    生成水印的遮罩
    :param image: 图像数据
    :param roi: 水印的ROI坐标和尺寸 (x, y, w, h)
    :return: 水印的遮罩图像数据，如果ROI未选择则返回 None
    """
    if roi is None:
        print("ROI not selected. Watermark removal aborted.")
        return None

    mask = detect_watermark_adaptive(image, roi)

    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(mask, kernel)


def remove_watermark(image_path, output_path):
    """
    去除图片中的水印
    :param image_path: 输入图像路径
    :param output_path: 输出图像路径
    :return: 处理后的图片
    """
    # 读取图像
    image = cv2.imread(image_path,)

    # 拷贝一份用来框选位置
    image_toSelect = image.copy()

    # 选择水印的ROI
    roi = select_roi_for_mask(image_toSelect)

    # 生成水印遮罩
    watermark_mask = generate_watermark_mask(image, roi)

    # 如果没有选择ROI，则不进行处理
    if roi is None or watermark_mask is None:
        return

    # 应用遮罩去除水印
    result_image = cv2.inpaint(image, watermark_mask, 3, cv2.INPAINT_NS)

    # 保存结果
    cv2.imwrite(output_path, result_image)

    print("Successfully removed watermark and saved result.")

    return result_image


if __name__ == "__main__":
    # input_image_path = "Images/DogFace_Watermark.jpg"
    # output_image_path = "Images/DogFace_Watermark_ToRemove.jpg"

    input_image_path = "Images/Meeting1.webp"
    output_image_path = "Images/Meeting1_new.webp"

    remove_watermark(input_image_path, output_image_path)
