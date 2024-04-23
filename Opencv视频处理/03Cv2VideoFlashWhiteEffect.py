"""
视频闪白效果
    1、读取视频: 使用OpenCV库中的cv2.VideoCapture()函数读取视频文件，获取视频的帧率、尺寸等信息。
    2、处理每一帧: 遍历视频的每一帧，对每一帧进行闪白处理。闪白处理通常有以下几种方法：
    3、将特定帧的所有像素值设置为白色。
    对特定帧进行增白处理，例如应用伽马变换使图像变亮。
    写入输出视频: 将处理后的每一帧写入一个新的视频文件中，形成闪白效果的视频。
"""


import cv2
import numpy as np


def flash_white(frame, frame_index):
    """
    闪白处理 1
    :param frame:
    :param frame_index:
    :return:
    """
    if frame_index < 5 or frame_index % 5 == 0:
        return 255 * np.ones_like(frame, dtype=np.uint8)
    else:
        return frame


def flash_white2(frame, frame_index):
    """
    闪白处理 2
    :param frame:
    :param frame_index:
    :return:
    """
    if frame_index < 5 or frame_index % 5 == 0:
        return gamma_trans(frame, 0.03)
    else:
        return frame


def gamma_trans(img, gamma):
    """
    增白处理
    :param img:
    :param gamma:
    :return:
    """
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def main():
    video_path = "Videos/CatRun.mp4"
    output_path = "Videos/VideoFlashWhite.mp4"

    cap = cv2.VideoCapture(video_path)

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 闪白效果
        frame = flash_white2(frame, frame_index)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
