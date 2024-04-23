"""
简单指定视频某片段重复播放效果
    1、使用 OpenCV 中的 cv2.VideoCapture() 函数读取原始视频文件。
    2、获取原始视频的帧率、总帧数和尺寸等信息。
    3、对指定的重复片段之前、重复片段和重复片段之后的部分进行逐帧处理。
    4、利用 cv2.VideoWriter() 创建输出视频文件对象，并按照顺序将处理好的视频帧写入其中。
"""

import cv2


def repeat_video_segment(input_video_path, output_video_path, start_frame, end_frame, repeat_count):
    """
    指定视频片段重复若干次
    :param input_video_path:
    :param output_video_path:
    :param start_frame:
    :param end_frame:
    :param repeat_count:
    :return:
    """
    # 读取原始视频文件
    cap = cv2.VideoCapture(input_video_path)

    # 获取原始视频的帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 校验起始帧和结束帧是否有效
    if start_frame < 0 or end_frame < 0 or start_frame >= total_frames or end_frame >= total_frames:
        print("Error: Invalid start or end frame.")
        return

    # 获取原始视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建输出视频文件对象
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 设置视频文件读取的起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 写入重复片段之前的部分
    for frame_index in range(start_frame):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    # 写入重复片段
    for _ in range(repeat_count):
        # 设置视频文件读取的起始帧为重复片段的起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 逐帧读取重复片段并写入输出视频对象
        for frame_index in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break

    # 写入重复片段之后的部分
    for frame_index in range(end_frame + 1, total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    # 调用函数并指定输入和输出视频文件路径，以及要重复播放的视频段的起始帧和结束帧，以及重复播放的次数
    input_video_path = "Videos/CatRun.mp4"
    output_video_path = "Videos/VideoRepeatSegment.mp4"
    start_frame = 130  # 重复片段的起始帧
    end_frame = 200  # 重复片段的结束帧
    repeat_count = 3  # 重复播放的次数
    repeat_video_segment(input_video_path, output_video_path, start_frame, end_frame, repeat_count)


if __name__ == "__main__":
    main()
