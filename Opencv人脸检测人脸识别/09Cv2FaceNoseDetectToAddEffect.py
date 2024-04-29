"""
简单进行鼻子检测并添加特效的功能实现
    1、加载人脸、眼睛和鼻子检测器以及鼻子特效图片。
    2、打开视频文件，并循环读取视频的每一帧。
    3、将每一帧转换为灰度图像。
    4、在灰度图像上检测人脸，并遍历每个检测到的人脸。
    5、对于每个人脸，通过检测眼睛并计算眼睛中心线的斜率来确定鼻子特效的旋转角度。
    6、在检测到的人脸区域内检测鼻子，并对每个检测到的鼻子进行处理。
    7、将旋转后的鼻子特效叠加在原始视频帧上。
    8、显示处理后的视频帧，并等待用户按下键盘上的“q”键退出。
"""

import cv2
import numpy as np
import os


def detect_and_add_nose_effect(video_path, nose_effect_image_path):
    """
    通过检测人脸、眼睛和鼻子，给视频中的人脸添加鼻子特效
    :param video_path: 视频文件路径
    :param nose_effect_image_path: 鼻子特效图片文件路径
    :return:
    """
    # 检查视频文件路径是否存在
    if not os.path.exists(video_path):
        print("Error: 视频文件路径不存在！")
        return

    # 检查鼻子特效图片文件路径是否存在
    if not os.path.exists(nose_effect_image_path):
        print("Error: 鼻子特效图片文件路径不存在！")
        return

    # 加载人脸、眼睛和鼻子检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    # nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml') # 系统没有该数据，注释使用下面的
    nose_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_mcs_nose.xml')

    # 加载鼻子特效图片
    nose_effect = cv2.imread(nose_effect_image_path, cv2.IMREAD_UNCHANGED)

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
                angle = -angle  # 取反，适应后面鼻子的旋转

                # 在画面上绘制眼睛的位置
                for (ex, ey, ew, eh) in eyes:
                    # cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                    print("cv2.rectangle eye ")

            # 鼻子检测
            noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=2.5, minNeighbors=15, minSize=(20, 20))
            for (nx, ny, nw, nh) in noses:
                # 在人脸区域绘制鼻子矩形框
                # cv2.rectangle(frame, (x+nx, y+ny), (x+nx+nw, y+ny+nh), (255, 255, 0), 2)

                # 调整鼻子特效图片的大小以适应鼻子区域
                resized_nose_effect = cv2.resize(nose_effect, (nw, nh))

                # 计算鼻子区域的中心点和旋转中心
                nose_center = (int(x + nx + nw // 2), int(y + ny + nh // 2))
                rotation_center = (int((nx + nw) // 2), int((ny + nh) // 2))

                # 旋转鼻子特效图片
                nose_effect_rotated = cv2.warpAffine(resized_nose_effect,
                                                     cv2.getRotationMatrix2D(rotation_center, angle, 1), (nw, nh))

                # 在鼻子位置上添加旋转后的特效图片
                for c in range(0, 3):
                    frame[y + ny:y + ny + nh, x + nx:x + nx + nw, c] = \
                        nose_effect_rotated[:, :, c] * (nose_effect_rotated[:, :, 3] / 255.0) + \
                        frame[y + ny:y + ny + nh, x + nx:x + nx + nw, c] * (1.0 - nose_effect_rotated[:, :, 3] / 255.0)

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
    nose_effect_image_path = 'Images/pig_nose_flat.png'
    detect_and_add_nose_effect(video_path, nose_effect_image_path)


if __name__ == "__main__":
    main()
