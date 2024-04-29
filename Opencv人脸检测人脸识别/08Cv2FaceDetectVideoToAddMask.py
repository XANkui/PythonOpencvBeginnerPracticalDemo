"""
简单进行人脸检测并添加面具特效实现
    1、人脸检测：

        使用OpenCV的Haar级联分类器来检测视频帧中的人脸。
        首先将视频帧转换为灰度图像，因为Haar级联分类器对灰度图像的效果更好。
        然后使用detectMultiScale方法在灰度图像中检测人脸，返回每个检测到的人脸的位置和尺寸。

    2、面具叠加：

        对于每个检测到的人脸，计算人脸的中心坐标和尺寸。
        根据人脸的中心坐标和尺寸，在人脸的位置上叠加面具。
        面具的大小会根据人脸的尺寸进行调整，以确保与人脸尺寸匹配。
        如果检测到多个人脸，将为每个人脸叠加面具。

    3、面具位置调整：

        通过按键控制面具在视频帧上的位置。
        按下键盘上的"W"、"A"、"S"、"D"键，分别向上、向左、向下、向右移动面具。

    4、角度计算：

        尝试检测人脸区域中的眼睛位置。
        如果检测到两只眼睛，则计算两只眼睛的中心坐标，并根据它们的位置计算人脸的偏转角度。
        这个角度可以用来调整面具的旋转，使其与人脸的角度保持一致。
        角度计算是根据两只眼睛的位置确定的，因为眼睛通常是人脸的重要特征，角度计算的准确性取决于眼睛检测的准确性。
"""

import cv2
import numpy as np


def resize_mask(mask, face_width, face_height):
    """
    调整面具的大小以适应人脸的尺寸
    :param mask: numpy 数组，表示原始面具图像
    :param face_width: float，表示人脸的宽度
    :param face_height: float，表示人脸的高度
    :return: 调整后的面具图像
    """
    if mask is None or face_width <= 0 or face_height <= 0:
        return None

    mask_height, mask_width = mask.shape[:2]
    mask = cv2.resize(mask, (int(face_width), int(face_height)))
    return mask


def overlay_mask(mask, frame, face_x, face_y, face_width, face_height, offset_x=0, offset_y=0, angle=0):
    """
    将面具叠加到视频帧上
    :param mask: numpy 数组，表示面具图像
    :param frame: numpy 数组，表示视频帧图像
    :param face_x: float，表示人脸中心点的 x 坐标
    :param face_y: float，表示人脸中心点的 y 坐标
    :param face_width: float，表示人脸的宽度
    :param face_height: float，表示人脸的高度
    :param offset_x: 可选参数，表示在 x 方向上的偏移量，默认为0
    :param offset_y: 可选参数，表示在 y 方向上的偏移量，默认为0
    :param angle: 可选参数，表示旋转角度，默认为0
    :return: 叠加了面具的视频帧图像
    """
    if mask is None or frame is None or face_width <= 0 or face_height <= 0:
        return None

    # 获取面具的大小和位置
    mask_height, mask_width = mask.shape[:2]

    # 计算旋转后的面具图像
    mask_center = (mask_width // 2, mask_height // 2)
    mask_matrix = cv2.getRotationMatrix2D(mask_center, angle, 1)
    rotated_mask = cv2.warpAffine(mask, mask_matrix, (mask_width, mask_height))

    # 获取旋转后的面具覆盖的区域
    mask_y = int(face_y - face_height / 2 + offset_y)
    mask_y_end = int(mask_y + face_height)
    mask_x = int(face_x - face_width / 2 + offset_x)
    mask_x_end = int(mask_x + face_width)

    # 确保旋转后的面具图像在视频帧内
    mask_x_start = max(mask_x, 0)
    mask_x_end_clip = min(mask_x_end, frame.shape[1])
    mask_y_start = max(mask_y, 0)
    mask_y_end_clip = min(mask_y_end, frame.shape[0])

    mask_width_clip = mask_x_end_clip - mask_x_start
    mask_height_clip = mask_y_end_clip - mask_y_start

    mask_x_end_local = mask_width_clip + max(mask_x - mask_x_start, 0)
    mask_y_end_local = mask_height_clip + max(mask_y - mask_y_start, 0)

    mask_x_local = max(mask_x - mask_x_start, 0)
    mask_y_local = max(mask_y - mask_y_start, 0)

    # 调整面具大小
    resized_mask = resize_mask(rotated_mask, mask_width_clip, mask_height_clip)

    # 将面具叠加到视频帧上
    for c in range(0, 3):
        try:
            frame[mask_y_start:mask_y_end_clip, mask_x_start:mask_x_end_clip, c] = \
                resized_mask[mask_y_local:mask_y_end_local, mask_x_local:mask_x_end_local, c] * (
                            resized_mask[mask_y_local:mask_y_end_local, mask_x_local:mask_x_end_local, 3] / 255.0) + \
                frame[mask_y_start:mask_y_end_clip, mask_x_start:mask_x_end_clip, c] * (
                            1.0 - resized_mask[mask_y_local:mask_y_end_local, mask_x_local:mask_x_end_local, 3] / 255.0)
        except ValueError:
            pass

    return frame


def detect_faces(frame):
    """
    检测视频帧中的人脸
    :param frame: numpy 数组，表示视频帧图像
    :return: 包含检测到的人脸信息的列表
    """
    if frame is None:
        return []

    # 使用人脸检测器检测人脸
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 获取人脸区域中的眼睛位置
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.35, minNeighbors=15, minSize=(20, 20))
        if len(eyes) >= 2:
            # 计算人脸的偏转角度
            eye_centers = np.array([(x + ex + ew // 2, y + ey + eh // 2) for (ex, ey, ew, eh) in eyes])
            angle = np.arctan2(eye_centers[1][1] - eye_centers[0][1], eye_centers[1][0] - eye_centers[0][0]) * 180 / np.pi

            angle = np.clip(angle, -45, 45)  # 将角度限制在[-45, 45]范围内
            print("angle = " + str(angle))
            return faces, angle
    return faces, 0


def main():
    # 加载面具图像
    mask = cv2.imread('Images/Cat_FaceMask.png', cv2.IMREAD_UNCHANGED)

    # 打开视频文件
    cap = cv2.VideoCapture('Videos/GirlFace.mp4')

    # 初始化面具位置偏移
    offset_x = 0
    offset_y = -65

    while True:
        # 读取一帧视频
        ret, frame = cap.read()
        if not ret:
            break

        # 检测人脸和角度
        faces, angle = detect_faces(frame)

        # 对每张人脸应用面具
        for (x, y, w, h) in faces:
            # 将面具叠加到人脸上
            # frame = overlay_mask(mask, frame, x + w // 2, y + h // 2, w, h, offset_x, offset_y, angle) # 这里变化太大，与视频不太符合，暂时不用
            frame = overlay_mask(mask, frame, x + w // 2, y + h // 2, w, h, offset_x, offset_y)

        # 显示结果
        cv2.imshow('Masked Faces', frame)

        # 检测按键输入
        key = cv2.waitKey(1)
        if key == ord('q'):  # 按 'q' 键退出
            break
        elif key == ord('w'):  # 上移面具
            offset_y -= 5
        elif key == ord('s'):  # 下移面具
            offset_y += 5
        elif key == ord('a'):  # 左移面具
            offset_x -= 5
        elif key == ord('d'):  # 右移面具
            offset_x += 5

    # 释放视频捕获对象
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
