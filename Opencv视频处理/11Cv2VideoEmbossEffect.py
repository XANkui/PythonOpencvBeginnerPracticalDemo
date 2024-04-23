"""
简单视频浮雕画效果
    1、打开视频文件：使用OpenCV的VideoCapture函数打开输入视频文件。
    2、设置输出视频参数：获取输入视频的帧率和尺寸，并定义输出视频的编码器和帧率。
    3、应用浮雕效果：定义一个apply_emboss_filter函数，该函数接受一帧图像作为输入，并应用浮雕效果。浮雕效果的实现基于以下步骤：
    4、写入输出视频：将处理后的帧写入输出视频文件。
    5、释放资源：释放所有使用的资源，包括输入视频和输出视频。
"""


import cv2


def apply_emboss_filter(frame, scale_factor=0.5, offset=128):
    """
    应用浮雕滤波器
    :param frame:
    :param scale_factor:
    :param offset:
    :return:
    """
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 通过对灰度图像应用 Sobel 算子来计算图像的梯度
    sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_8U, 0, 1, ksize=3)

    # 计算浮雕效果
    emboss = cv2.addWeighted(sobel_x, scale_factor, sobel_y, scale_factor, offset)

    # 将浮雕效果转换回 BGR 格式
    emboss_bgr = cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)

    return emboss_bgr


def apply_emboss_effect(input_video_path, output_video_path, scale_factor=0.5, offset=128):
    """
    应用浮雕效果到整个视频
    :param input_video_path:
    :param output_video_path:
    :param scale_factor:
    :param offset:
    :return:
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Failed to open video.")
        return

    # 获取视频的帧率和尺寸
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        # 读取视频的一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 对当前帧应用浮雕滤波器
        embossed_frame = apply_emboss_filter(frame, scale_factor, offset)

        # 将处理后的帧写入输出视频文件
        out.write(embossed_frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    # 调用函数并指定输入和输出视频文件路径
    input_video_path = "Videos/TwoPeopleRunning.mp4"
    output_video_path = "Videos/VideoEmbossEffect.mp4"
    apply_emboss_effect(input_video_path, output_video_path, scale_factor=0.5, offset=128)


if __name__ == "__main__":
    main()






