"""
简单动态聚光灯效果
    1、apply_spotlights(frame, spotlights):
    这个函数用于在视频帧上应用多盏聚光灯效果。
    参数：
        frame：输入的视频帧，是一个 numpy 数组。
        spotlights：包含多盏聚光灯信息的列表。每个聚光灯由位置、颜色、移动角度和速度组成。
    返回值：
        处理后的视频帧，应用了聚光灯效果。

    2、main():
    这个函数是程序的主函数，用于读取输入视频并逐帧处理，添加聚光灯效果后写入输出视频。
    主要步骤：
        读取输入视频文件。
        初始化输出视频文件。
        创建并初始化多盏聚光灯的信息。
        逐帧读取输入视频，应用聚光灯效果并写入输出视频。
        更新每盏聚光灯的位置。
    函数调用：
    调用了apply_spotlights()函数来添加聚光灯效果。
"""

import cv2
import numpy as np
import random
import math


def apply_spotlights(frame, spotlights):
    """
    在视频帧上应用多盏聚光灯效果
    :param frame:
    :param spotlights:
    :return:
    """
    result_frame = frame.copy()

    # 在每一盏聚光灯上叠加光斑
    for spotlight in spotlights:
        spotlight_pos, spotlight_color, _, _ = spotlight  # 保留移动方向和速度，但在此不使用
        spotlight_mask = np.zeros_like(frame)
        cv2.ellipse(spotlight_mask, spotlight_pos, (150, 150), 0, 0, 360, spotlight_color, -1)
        result_frame = cv2.addWeighted(result_frame, 1, spotlight_mask, 0.5, 0)

    return result_frame


def main():
    video_path = "Videos/CatRun.mp4"
    output_path = "Videos/VideoSpotLightEffect.mp4"

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 初始化聚光灯列表，每一盏聚光灯由位置、颜色、移动角度和速度组成
    spotlights = []
    for _ in range(5):  # 创建5盏聚光灯
        spotlight_pos = (random.randint(0, width - 1), random.randint(0, height - 1))
        spotlight_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # 随机选择一个移动角度和速度
        angle = random.uniform(0, 2 * math.pi)
        speed = random.randint(3, 10)
        spotlights.append((spotlight_pos, spotlight_color, angle, speed))

    # 逐帧处理视频
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 应用多盏聚光灯效果
        frame_with_spotlights = apply_spotlights(frame, spotlights)

        # 写入输出视频
        out.write(frame_with_spotlights)

        # 更新每一盏聚光灯的位置
        for i in range(len(spotlights)):
            # 获取当前聚光灯的位置、颜色、移动角度和速度
            spotlight_pos, spotlight_color, angle, speed = spotlights[i]
            # 根据移动角度和速度更新聚光灯位置
            dx = int(speed * math.cos(angle))
            dy = int(speed * math.sin(angle))
            new_x = min(max(0, spotlight_pos[0] + dx), width - 1)
            new_y = min(max(0, spotlight_pos[1] + dy), height - 1)
            # 如果聚光灯到达视频边缘，随机选择一个新的移动角度和速度
            if new_x in [0, width - 1] or new_y in [0, height - 1]:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.randint(3, 10)
            # 更新聚光灯列表中的聚光灯信息
            spotlights[i] = ((new_x, new_y), spotlight_color, angle, speed)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
