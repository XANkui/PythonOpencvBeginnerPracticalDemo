"""
单个聚光等在视频中移动的效果
"""

import cv2
import numpy as np


def apply_spotlight(frame, spotlight_pos):
    """
    在视频帧上应用动态聚光灯效果
    :param frame:
    :param spotlight_pos:
    :return:
    """
    # 创建一个黑色图像，与原始视频帧相同大小
    spotlight_mask = np.zeros_like(frame)

    # 在黑色图像上绘制一个白色的椭圆，模拟聚光灯光圈
    cv2.ellipse(spotlight_mask, spotlight_pos, (100, 100), 0, 0, 360, (255, 255, 255), -1)

    # 将光圈图像与原始视频帧进行叠加
    result_frame = cv2.addWeighted(frame, 1, spotlight_mask, 0.5, 0)

    return result_frame


def main():
    # 打开视频文件
    cap = cv2.VideoCapture('Videos/CatRun.mp4')

    # 获取视频帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    # 初始化聚光灯位置
    spotlight_pos = (int(width / 2), int(height / 2))

    # 逐帧处理视频
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 应用动态聚光灯效果
        frame_with_spotlight = apply_spotlight(frame, spotlight_pos)

        # 写入输出视频
        out.write(frame_with_spotlight)

        # 更新聚光灯位置（示例中简单地沿着视频宽度方向移动）
        spotlight_pos = ((spotlight_pos[0] + 5) % width, spotlight_pos[1])

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
