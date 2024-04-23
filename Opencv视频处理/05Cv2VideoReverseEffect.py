"""
简单视频倒放效果
    1、读取视频文件，并获取视频的帧率、总帧数、宽度和高度等信息。
    2、创建一个空的视频写入对象，用于保存倒放后的视频。
    3、从视频的最后一帧开始，逐帧读取视频并写入新的视频对象，直到第一帧。
    4、保存并关闭新的视频文件。
"""

import cv2


def reverse_video(input_video_path, output_video_path):
    """
    视频倒序保存
    :param input_video_path:
    :param output_video_path:
    :return:
    """
    # 读取视频文件
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

    # 逐帧读取视频并写入新的视频对象
    for frame_index in range(total_frames - 1, -1, -1):
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
    # 调用函数并指定输入和输出视频文件路径
    input_video_path = "Videos/CatRun.mp4"
    output_video_path = "Videos/VideoReverse.mp4"
    reverse_video(input_video_path, output_video_path)


if __name__ == "__main__":
    main()
