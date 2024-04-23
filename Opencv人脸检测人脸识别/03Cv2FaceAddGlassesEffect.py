"""
简单人脸检测添加戴眼镜效果
    1、使用 OpenCV 加载人脸识别分类器。
    2、读取人脸图像和眼镜图像。
    3、对人脸图像进行人脸检测，获取人脸的位置信息。
    4、遍历检测到的每张人脸，根据人脸宽度调整眼镜大小。
    5、将调整后的眼镜图像覆盖到对应人脸位置上。
    6、返回带有眼镜效果的图像数据。
"""

import cv2


def over_img(img, img_over, over_x, over_y):
    """
    将一张图像覆盖到另一张图像上
    :param img: (numpy.ndarray) 目标图像数据
    :param img_over: (numpy.ndarray) 待覆盖图像数据，包含 alpha 通道
    :param over_x: (int) 待覆盖图像左上角的 x 坐标
    :param over_y: (int) 待覆盖图像左上角的 y 坐标
    :return: numpy.ndarray 覆盖后的图像数据
    """
    img_h, img_w, c = img.shape
    img_over_h, img_over_w, over_c = img_over.shape
    # 将待覆盖图像转换为带 alpha 通道的 BGRA 格式
    if over_c == 3:
        img_over = cv2.cvtColor(img_over, cv2.COLOR_BGR2BGRA)
    # 遍历待覆盖图像的每个像素
    for w in range(0, img_over_w):
        for h in range(0, img_over_h):
            # 透明像素不能覆盖目标图像
            if img_over[h, w, 3] != 0:
                # 遍历 RGB 通道
                for c in range(0, 3):
                    x = over_x + w
                    y = over_y + h
                    # 如果超出目标图像范围，则跳出循环
                    if x >= img_w or y >= img_h:
                        break
                    # 将待覆盖图像像素覆盖到目标图像上
                    img[y, x, c] = img_over[h, w, c]
    return img


def apply_glasses(input_image_path, glasses_image_path, vertical_offset=0.35):
    """
    在人脸图像上添加眼镜效果
    :param input_image_path: (str) 输入的人脸图像路径
    :param glasses_image_path: (str) 眼镜图像的路径
    :param vertical_offset: (float) 眼镜垂直位置的调整参数，范围为0到1，默认值为0.35
    :return: numpy.ndarray 带眼镜效果的图像数据
    """
    # 参数安全性校验
    if not isinstance(input_image_path, str) or not input_image_path.strip():
        raise ValueError("Invalid input image path.")

    if not isinstance(glasses_image_path, str) or not glasses_image_path.strip():
        raise ValueError("Invalid glasses image path.")

    if not (0 <= vertical_offset <= 1):
        raise ValueError("Vertical offset parameter must be between 0 and 1.")

    # 读取人脸和眼镜图像
    img = cv2.imread(input_image_path)
    glass = cv2.imread(glasses_image_path, cv2.IMREAD_UNCHANGED)  # 保留图像类型
    height, weight, channel = glass.shape
    # 加载人脸识别联结器
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # 进行人脸检测
    faces = faceCascade.detectMultiScale(img, 1.15, 4)
    # 对每个检测到的人脸应用眼镜效果
    for (x, y, w, h) in faces:
        gw = w
        gh = int(height * gw / weight)
        # 调整眼镜图像大小以适应人脸宽度
        img_over_new = cv2.resize(glass, (gw, gh))
        # 将眼镜图像覆盖到人脸图像上
        img = over_img(img, img_over_new, x, y + int(h * vertical_offset))
        # 绘制脸部范围图框
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
    return img


# 测试接口调用
if __name__ == "__main__":
    input_image_path = "Images/TwoManFace.png"
    glasses_image_path = "Images/glasses.png"

    try:
        output_img = apply_glasses(input_image_path, glasses_image_path, vertical_offset=0.0)
        cv2.imshow("output_img", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Glasses applied successfully.")
    except ValueError as ve:
        print(f"Error: {ve}")
