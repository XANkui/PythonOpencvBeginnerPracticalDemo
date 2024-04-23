"""
简单人脸眼睛检测添加睫毛效果
    1、加载人脸和眼睛检测器。
    2、读取输入的人脸图像，并对其进行灰度化处理。
    3、使用人脸检测器检测人脸区域。
    4、对每个检测到的人脸区域，使用眼睛检测器检测眼睛。
    5、对每个检测到的眼睛，判断其是否为左眼，并获取其位置信息。
    6、如果是左眼，根据竖直方向的偏移参数调整睫毛的位置。
    7、将调整后的睫毛图像合成到原始图像中。
    8、返回添加了睫毛效果的图像。
"""

import cv2


def apply_eyelash_left_eye(image_path, eyelash_image_path, vertical_offset=0.0):
    """
    在人脸图像上添加左眼睫毛效果
    :param image_path: (str) 输入的人脸图像路径
    :param eyelash_image_path: (str) 睫毛图像的路径
    :param vertical_offset: (float) 睫毛竖直方向位置调节参数，范围为-1到1，默认值为0（中心位置）
    :return: numpy.ndarray 添加了睫毛效果的图像数据
    """

    # 参数安全性校验
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("Invalid input image path.")
    if not isinstance(eyelash_image_path, str) or not eyelash_image_path.strip():
        raise ValueError("Invalid eyelash image path.")
    if not (-1 <= vertical_offset <= 1):
        raise ValueError("Vertical offset parameter must be between -1 and 1.")

    # 读取图像
    image = cv2.imread(image_path)
    eyelash_image = cv2.imread(eyelash_image_path, cv2.IMREAD_UNCHANGED)

    # 加载人脸和眼睛检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # 灰度化图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 对每个人脸进行操作
    for (x, y, w, h) in faces:
        # 提取人脸区域
        face_roi = image[y:y + h, x:x + w]
        gray_roi = gray[y:y + h, x:x + w]

        # 检测眼睛
        eyes = eye_cascade.detectMultiScale(gray_roi)

        # 画出左右眼范围的矩形框
        for (ex, ey, ew, eh) in eyes:
            eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

            # 仅对左眼进行操作
            if eye_center[0] < x + w // 2:
                # 调整睫毛图像大小以匹配眼睛
                resized_eyelash = cv2.resize(eyelash_image, (ew, eh))

                # 调整睫毛竖直方向位置
                offset_y = int(eh * vertical_offset)
                ey = max(0, ey + offset_y)

                # 将睫毛图像合成到原始图像中
                for c in range(0, 3):
                    for i in range(eh):
                        for j in range(ew):
                            if resized_eyelash[i, j, 3] != 0:
                                image[y + ey + i, x + ex + j, c] = resized_eyelash[i, j, c]

    # 返回添加了左眼睫毛效果的图像
    return image


# 测试接口调用
if __name__ == "__main__":
    input_image_path = "Images/ManWoman.jpeg"
    eyelash_image_path = "Images/eyelash.png"

    try:
        output_img = apply_eyelash_left_eye(input_image_path, eyelash_image_path, vertical_offset=-0.1)
        cv2.imshow("Eyelash Effect", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Eyelash applied successfully.")
    except ValueError as ve:
        print(f"Error: {ve}")
