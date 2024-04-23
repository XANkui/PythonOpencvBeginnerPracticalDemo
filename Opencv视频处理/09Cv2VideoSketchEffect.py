"""
简单视频素描效果
    1、读取视频：首先，我们使用OpenCV库读取输入的视频文件。
    2、处理每一帧：对于视频中的每一帧，我们都会应用素描画效果。
    我们使用边缘检测算法（如Canny边缘检测）来检测图像中的边缘，然后反转边缘图像，使其成为黑色背景上的白色轮廓。
    3、保存视频：最后，我们将处理后的帧写入输出视频文件中，以创建包含素描效果的新视频。
"""

import cv2


def sketch(frame, canny_threshold=150):
    """
    简单素描效果
    :param frame:帧，图
    :param canny_threshold:边缘因子，越大，细节越少
    :return:
    """
    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊去除噪音
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 检测边缘
    edges = cv2.Canny(blurred, canny_threshold, canny_threshold * 3)
    # 反色
    edges = 255 - edges
    return edges


def sketch_video(input_video_path, output_video_path, canny_threshold=150):
    """
    简单视频素描效果
    :param input_video_path:
    :param output_video_path:
    :param canny_threshold: 边缘因子，越大，细节越少
    :return:
    """
    # 校验视频路径
    if not isinstance(input_video_path, str) or not isinstance(output_video_path, str):
        raise ValueError("Input and output video paths must be strings.")
    # 读取原始视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Error: Unable to open input video.")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 校验视频尺寸
    if width <= 0 or height <= 0:
        raise ValueError("Error: Invalid video dimensions.")

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

    # 处理每一帧并保存视频
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 应用素描画效果
        sketch_frame = sketch(frame, canny_threshold)

        # 写入视频帧
        out.write(sketch_frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    # 调用函数并指定输入和输出视频文件路径
    input_video_path = "Videos/CatRun.mp4"
    output_video_path = "Videos/VideoSketchEffect.mp4"
    sketch_video(input_video_path, output_video_path, canny_threshold=7)


if __name__ == "__main__":
    main()
