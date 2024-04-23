"""
简单行人人体检测效果
    1、加载 Haar 级联分类器用于行人检测。
    2、使用 VideoCapture 对象从给定的视频文件中读取帧。
    3、将每一帧转换为灰度图像，并使用 detectMultiScale 函数检测行人。
    4、对于每个检测到的行人，使用 rectangle 函数绘制一个矩形框。
    5、循环遍历直到视频结束或者按下 'q' 键退出，然后释放资源。
"""

import cv2


def detect_and_draw_pedestrians(video_path, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    检测视频中的行人并绘制矩形框
    :param video_path: (str) 输入视频文件的路径
    :param scale_factor: (float) 检测窗口的缩放比例，默认为1.1
    :param min_neighbors: (int) 每个候选矩形应该保留的邻近矩形的数量阈值，默认为5
    :param min_size: (tuple) 行人矩形的最小尺寸，格式为(w, h)，默认为(30, 30)
    :return:
    """
    # 参数安全校验
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("Invalid video path.")
    if not isinstance(scale_factor, (int, float)) or scale_factor <= 1.0:
        raise ValueError("Scale factor must be a positive float greater than 1.0.")
    if not isinstance(min_neighbors, int) or min_neighbors < 0:
        raise ValueError("Min neighbors must be a non-negative integer.")
    if not isinstance(min_size, tuple) or len(min_size) != 2 or not all(isinstance(val, int) for val in min_size) or \
            min_size[0] <= 0 or min_size[1] <= 0:
        raise ValueError("Min size must be a tuple of two positive integers.")

    # 加载行人检测器
    pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    # 读取视频文件
    video_capture = cv2.VideoCapture(video_path)

    while True:
        # 读取一帧视频
        ret, frame = video_capture.read()

        if not ret:
            break

        # 将图像转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测行人
        pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                                          minSize=min_size)

        # 绘制矩形框
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Pedestrian Detection', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频捕获对象
    video_capture.release()
    cv2.destroyAllWindows()


# 测试接口函数
if __name__ == "__main__":
    video_path = 'Videos/TwoPeopleRunning.mp4'
    detect_and_draw_pedestrians(video_path, scale_factor=1.2, min_neighbors=3, min_size=(50, 50))
