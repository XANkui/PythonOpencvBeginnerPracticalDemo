"""
简单视频抖动放大效果

    apply_zoom() 函数：
    这个函数接受一个帧图像和放大因子作为参数，然后将图像放大。
    首先，计算了放大区域的左上角和右下角坐标，然后从原图中截取出放大区域，最后将截取的图像进行放大并返回。

    video_shake() 函数：
    这个函数接受视频文件的路径和输出视频文件的路径作为参数。
    首先，打开视频文件并获取视频信息，包括帧率、宽度和高度。
    然后，创建一个 VideoWriter 对象，用于写入输出视频文件。
    在一个循环中，逐帧读取视频，检查帧索引是否为前 5 帧或者 10 的倍数帧，如果是，则应用放大效果。
    最后，将处理后的帧写入输出视频文件，释放资源。
"""

import cv2


def apply_zoom(frame, factor):
    """
    放大指定帧图
    :param frame:
    :param factor: 这个函数接受一个帧图像和放大因子作为参数，然后将图像放大。
    :return:
    """
    height, width, _ = frame.shape
    h1 = int(height * 0.1)
    h2 = int(height * 0.9)
    w1 = int(width * 0.1)
    w2 = int(width * 0.9)
    zoomed_frame = frame[h1:h2, w1:w2]
    return cv2.resize(zoomed_frame, (width, height))


def video_shake(video_path, output_path):
    """
    视频放大抖动
    :param video_path:
    :param output_path:
    :return:
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 视频索引
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 检查帧索引是否为 10 的倍数或前 5 帧
        if frame_index < 5 or frame_index % 10 == 0:
            frame = apply_zoom(frame, factor=2)

        out.write(frame)

        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    video_path = "Videos/CatRun.mp4"
    output_path = "Videos/VideoShake.mp4"

    video_shake(video_path, output_path)


if __name__ == "__main__":
    main()
