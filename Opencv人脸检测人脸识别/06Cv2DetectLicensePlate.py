"""
简单进行车牌检测和识别
    1、使用 OpenCV 加载级联分类器，并调用 detectMultiScale 方法检测车牌区域。
    2、利用 Tesseract OCR 对车牌区域进行字符识别。
    3、绘制矩形框和在图像上显示识别结果。
"""

import cv2
import pytesseract


def detect_license_plate(image_path, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    使用级联分类器检测车牌并绘制矩形框
    :param image_path: (str) 输入图像的路径
    :param scale_factor: (float) 每次图像尺寸减小的比例，默认为1.1
    :param min_neighbors: (int) 每个候选矩形应该保留的邻居数，默认为5
    :param min_size: (tuple) 矩形的最小尺寸，默认为(30, 30)
    :return:
    """
    # 参数安全校验
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("Invalid image path.")

    if not isinstance(scale_factor, float) or scale_factor <= 1.0:
        raise ValueError("Scale factor must be a float greater than 1.0.")

    if not isinstance(min_neighbors, int) or min_neighbors <= 0:
        raise ValueError("Min neighbors must be a positive integer.")

    if not isinstance(min_size, tuple) or len(min_size) != 2 or min_size[0] <= 0 or min_size[1] <= 0:
        raise ValueError("Min size must be a tuple of two positive integers.")

    # 加载车牌检测器
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    # 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测车牌
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                            minSize=min_size)

    if len(plates) > 0:
        for (x, y, w, h) in plates:
            plate_img = gray[y:y + h, x:x + w]  # 裁剪出车牌区域
            plate_text = pytesseract.image_to_string(plate_img, config='--psm 8')  # 使用Tesseract OCR进行字符识别
            plate_text = plate_text.strip()  # 去除空白字符

            # 在原始图像上绘制车牌区域的矩形框
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 在图像上显示识别结果
            cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow("License Plate Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No license plate detected.")


# 测试车牌检测函数
if __name__ == "__main__":
    image_path = "Images/CarPlate.jpeg"
    detect_license_plate(image_path, scale_factor=1.077, min_neighbors=3)
