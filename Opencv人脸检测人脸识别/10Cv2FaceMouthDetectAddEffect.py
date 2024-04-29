"""
简单进行嘴巴检测并添加特效的功能实现
    1、使用 OpenCV 提供的 Haar 级联检测器进行人脸、眼睛和嘴巴检测。
    2、通过检测到的眼睛位置计算偏转角度，以使特效与眼睛保持一致的方向。
    3、根据检测到的嘴巴位置，将特效添加到合适的位置上。可以通过调整特效图片的大小和旋转来适应嘴巴的形状和方向。

"""


import cv2
import numpy as np
import os


def detect_and_add_mouth_effect(video_path, mouth_effect_image_path, mouth_scale_factor=2.5, mouth_min_neighbors=30, mouth_min_size=(20, 20),
                                 mouth_offset_x=0, mouth_offset_y=0):
    """
    通过检测人脸、眼睛和鼻子，在视频中给人脸添加嘴巴特效
    :param video_path: 视频文件路径
    :param mouth_effect_image_path: 嘴巴特效图片文件路径
    :param mouth_scale_factor: 嘴巴检测器的缩放因子
    :param mouth_min_neighbors: 嘴巴检测器的最小邻居数
    :param mouth_min_size: 嘴巴检测器的最小尺寸
    :param mouth_offset_x: 嘴巴特效在水平方向的偏移量
    :param mouth_offset_y: 嘴巴特效在垂直方向的偏移量
    :return:
    """
    # 检查视频文件路径是否存在
    if not os.path.exists(video_path):
        print("Error: 视频文件路径不存在！")
        return

    # 检查嘴巴特效图片文件路径是否存在
    if not os.path.exists(mouth_effect_image_path):
        print("Error: 嘴巴特效图片文件路径不存在！")
        return

    # 加载人脸、眼睛和鼻子检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_mcs_mouth.xml')

    # 加载嘴巴特效图片
    mouth_effect = cv2.imread(mouth_effect_image_path, cv2.IMREAD_UNCHANGED)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    while True:
        # 读取一帧视频
        ret, frame = cap.read()
        if not ret:
            break

        # 将视频帧转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 遍历检测到的人脸
        for (x, y, w, h) in faces:
            # 在人脸区域绘制矩形框
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # 在人脸区域检测眼睛
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.35, minNeighbors=15, minSize=(20, 20))
            if len(eyes) >= 2:
                # 计算两个眼睛的中心位置
                eye_centers = np.array([(x + ex + ew // 2, y + ey + eh // 2) for (ex, ey, ew, eh) in eyes])

                # 计算眼睛中心线的斜率
                slope = (eye_centers[1][1] - eye_centers[0][1]) / (eye_centers[1][0] - eye_centers[0][0])

                # 计算角度
                angle = np.arctan(slope) * 180 / np.pi
                angle = np.clip(angle, -45, 45)  # 将角度限制在[-45, 45]范围内
                angle = -angle  # 取反，适应后面特效的旋转

                # 在画面上绘制眼睛的位置
                for (ex, ey, ew, eh) in eyes:
                    # cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                    print("cv2.rectangle eye ")

            # 嘴巴检测
            mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=mouth_scale_factor, minNeighbors=mouth_min_neighbors, minSize=mouth_min_size)
            for (nx, ny, nw, nh) in mouth:
                # 在人脸区域绘制嘴巴矩形框
                # cv2.rectangle(frame, (x+nx, y+ny), (x+nx+nw, y+ny+nh), (255, 255, 0), 2)

                # 调整嘴巴特效图片的大小以适应嘴巴区域
                resized_mouth_effect = cv2.resize(mouth_effect, (nw, nh))

                # 计算嘴巴区域的中心点和旋转中心
                mouth_center = (int(x + nx + nw // 2) + mouth_offset_x, int(y + ny + nh // 2) + mouth_offset_y)
                rotation_center = (int((nx + nw) // 2), int((ny + nh) // 2))

                # 旋转嘴巴特效图片
                mouth_effect_rotated = cv2.warpAffine(resized_mouth_effect,
                                                       cv2.getRotationMatrix2D(rotation_center, angle, 1), (nw, nh))

                # 在嘴巴位置上添加旋转后的特效图片
                for c in range(0, 3):
                    frame[y + ny:y + ny + nh, x + nx:x + nx + nw, c] = \
                        mouth_effect_rotated[:, :, c] * (mouth_effect_rotated[:, :, 3] / 255.0) + \
                        frame[y + ny :y + ny + nh , x + nx:x + nx + nw, c] * (1.0 - mouth_effect_rotated[:, :, 3] / 255.0)

        # 显示结果
        cv2.imshow('Face and Features Detection', frame)

        # 检测按键输入
        key = cv2.waitKey(1)
        if key == ord('q'):  # 按 'q' 键退出
            break

    # 释放视频捕获对象
    cap.release()
    cv2.destroyAllWindows()


def main():
    # 使用示例
    video_path = 'Videos/GirlFace.mp4'
    mouth_effect_image_path = 'Images/mouth.png'
    detect_and_add_mouth_effect(video_path, mouth_effect_image_path, mouth_offset_x=10, mouth_offset_y=20)  # 例：向右移动10像素，向下移动20像素


if __name__ == "__main__":
    main()
