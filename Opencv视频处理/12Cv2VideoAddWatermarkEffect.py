"""
简单给视频添加水印图片效果
    1、打开输入视频文件，并获取其每一帧的大小。
    2、打开水印图像文件，并获取其大小。
    3、遍历视频的每一帧，将水印图像叠加到视频帧上。
    4、根据指定的位置参数确定水印在视频帧上的位置，并将水印图像叠加到相应的位置上。
    5、将带有水印的视频帧写入输出视频文件。
"""

import cv2
import os


def add_watermark_to_video(input_video_path, output_video_path, watermark_image_path, alpha=1.0,
                           position='bottom-right'):
    """
    简单给视频添加水印图片效果
    :param input_video_path: 原视频路径
    :param output_video_path: 添加水印后保存视频路径
    :param watermark_image_path: 水印图片路径
    :param alpha: 控制水印整体透明度的参数，范围为 0 到 1
    :param position: 控制水印位置的参数，可选值包括 'top-left', 'top-right', 'bottom-left', 'center', 'bottom-right'
    :return:
    """
    # 检查输入视频文件是否存在
    if not os.path.isfile(input_video_path):
        print("Error: Input video file does not exist.")
        return

    # 检查水印图像文件是否存在
    if not os.path.isfile(watermark_image_path):
        print("Error: Watermark image file does not exist.")
        return

    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Failed to open input video.")
        return

    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建VideoWriter对象用于写入输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    # 读取水印图像
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_UNCHANGED)

    # 检查水印图像是否成功读取
    if watermark is None:
        print("Error: Failed to read watermark image.")
        return

    # 检查水印图像的透明度通道是否存在
    if watermark.shape[2] < 4:
        print("Error: Watermark image does not have an alpha channel.")
        return

    # 循环遍历视频的每一帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将水印叠加到当前帧上
        overlay = frame.copy()
        h, w = watermark.shape[:2]

        # 根据位置参数确定水印的位置
        if position == 'top-left':
            x_offset, y_offset = 10, 10
        elif position == 'top-right':
            x_offset, y_offset = width - w - 10, 10
        elif position == 'bottom-left':
            x_offset, y_offset = 10, height - h - 10
        elif position == 'center':
            x_offset, y_offset = (width - w) // 2, (height - h) // 2
        else:  # 默认为 'bottom-right'
            x_offset, y_offset = width - w - 10, height - h - 10

        for c in range(0, 3):
            overlay[y_offset:y_offset + h, x_offset:x_offset + w, c] = \
                watermark[:, :, c] * (watermark[:, :, 3] / 255.0 * alpha) + frame[y_offset:y_offset + h,
                                                                            x_offset:x_offset + w, c] * (
                        1.0 - watermark[:, :, 3] / 255.0 * alpha)

        # 将帧写入输出视频
        out.write(overlay)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    # 调用函数并指定输入和输出视频文件路径以及水印图像路径
    input_video_path = "Videos/CatRun.mp4"
    output_video_path = "Videos/CatRun_Wartermark.mp4"
    watermark_image_path = "Images/Watermark.png"
    alpha = 0.5  # 控制水印整体透明度的参数，范围为 0 到 1
    position = 'top-right'  # 控制水印位置的参数，可选值包括 'top-left', 'top-right', 'bottom-left', 'center', 'bottom-right'

    # 执行函数
    add_watermark_to_video(input_video_path, output_video_path, watermark_image_path, alpha, position)


if __name__ == "__main__":
    main()
