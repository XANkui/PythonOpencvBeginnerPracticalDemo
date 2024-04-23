"""
简单指定视频某片段慢放效果
    1、读取视频文件。
    2、确定要添加慢放效果的帧范围。
    3、逐帧读取视频，并根据指定的帧范围和慢放速度，决定每一帧写入的次数。
    4、将处理后的帧写入新的视频文件。
"""

import cv2


def slow_motion(input_video_path, output_video_path, start_frame, end_frame, slow_down_factor):
    """
    指定视频指定帧段，进行慢放处理
    :param input_video_path:
    :param output_video_path:
    :param start_frame:
    :param end_frame:
    :param slow_down_factor:慢放因子，越大越慢
    :return:
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)

    # 检查视频文件是否成功打开
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # 获取视频帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_video_path, fourcc, fps / slow_down_factor, (width, height))

    # 校验起始帧和结束帧是否合法
    if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
        print("Error: Invalid start frame or end frame.")
        return

    # 校验慢放速度是否合法
    if slow_down_factor <= 0:
        print("Error: Slow down factor must be greater than 0.")
        return

    # 逐帧读取视频并添加慢放效果
    for frame_index in range(total_frames):
        ret, frame = cap.read()
        if ret:
            # 检查当前帧是否在指定的帧范围内
            if start_frame <= frame_index <= end_frame:
                # 写入当前帧多次，模拟慢放效果
                for _ in range(slow_down_factor):
                    out.write(frame)
            else:
                out.write(frame)
        else:
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    # 调用函数并指定输入和输出视频文件路径、帧范围和慢放速度
    input_video_path = "Videos/CatRun.mp4"
    output_video_path = "Videos/VideoSlowMotionEffect.mp4"
    start_frame = 150  # 起始帧
    end_frame = 160  # 结束帧
    slow_down_factor = 3  # 慢放速度
    slow_motion(input_video_path, output_video_path, start_frame, end_frame, slow_down_factor)


if __name__ == "__main__":
    main()
