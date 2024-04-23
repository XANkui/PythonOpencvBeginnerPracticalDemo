"""
简单指定视频某片段快放效果
    1、打开输入视频文件，获取视频信息。
    2、遍历视频的每一帧，根据指定的起始帧和结束帧以及速度因子确定写入帧的条件。
    3、写入正常播放的帧和指定帧段的帧。
"""

import cv2


def fast_forward_segment(input_video_path, output_video_path, start_frame, end_frame, speed_factor):
    """
    简单指定视频某片段快放效果
    :param input_video_path:
    :param output_video_path:
    :param start_frame:
    :param end_frame:
    :param speed_factor:快放因子，大于 0 的整数，越大播放越快
    :return:
    """
    # 打开输入视频文件
    cap = cv2.VideoCapture(input_video_path)

    # 获取视频帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 校验输入参数的合法性
    if start_frame < 0:
        start_frame = 0
    if end_frame > total_frames:
        end_frame = total_frames - 1
    if start_frame >= end_frame:
        print("Error: Invalid start and end frame.")
        return

    # 逐帧读取视频并写入新的视频对象
    for frame_index in range(total_frames):
        ret, frame = cap.read()
        if ret:
            # 写入正常播放的帧
            if frame_index < start_frame or frame_index > end_frame:
                out.write(frame)
            # 写入指定段的帧，并根据速度因子决定写入次数
            elif start_frame <= frame_index <= end_frame and (frame_index - start_frame) % speed_factor == 0:
                out.write(frame)
        else:
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    # 调用函数并指定输入和输出视频文件路径
    input_video_path = "Videos/CatRun.mp4"
    output_video_path = "Videos/VideoFastForwardEffect.mp4"
    start_frame = 100
    end_frame = 200
    speed_factor = 5  # 速度因子，决定指定帧段的帧被写入的次数间隔，可以调整
    fast_forward_segment(input_video_path, output_video_path, start_frame, end_frame, speed_factor)


if __name__ == "__main__":
    main()
