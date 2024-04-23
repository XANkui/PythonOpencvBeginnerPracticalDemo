"""
简单视频卡通画效果
    1、灰度化：首先将彩色图像转换为灰度图像，以便进行边缘检测。
    2、边缘检测：利用边缘检测算法（例如自适应阈值边缘检测）找到图像中的边缘部分，这些边缘部分将用于后续步骤。
    3、颜色量化：对彩色图像进行颜色量化，使得图像中的颜色变得更加平滑。这一步通常使用双边滤波器来实现。
    4、合并边缘和颜色图像：将边缘图像和颜色图像结合起来，只保留边缘部分对应的颜色。这样就得到了卡通效果的图像。
"""


import cv2


def cartoonize(image, edge_threshold=9, color_reduction=300):
    """
    图片卡通画效果
    :param image:
    :param edge_threshold:
    :param color_reduction:
    :return:
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, edge_threshold)

    # 颜色量化
    color = cv2.bilateralFilter(image, 9, color_reduction, color_reduction)

    # 合并边缘和颜色图像
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon


def main(input_video_path, output_video_path, edge_threshold=9, color_reduction=300):
    """
    简单视频卡通画效果
    :param input_video_path:
    :param output_video_path:
    :param edge_threshold:
    :param color_reduction:
    :return:
    """
    # 读取输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Unable to open input video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Error: Unable to create output video.")
        return

    # 逐帧处理视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 应用卡通画效果
        cartoon_frame = cartoonize(frame, edge_threshold, color_reduction)

        # 写入输出视频
        out.write(cartoon_frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    # 调用函数并指定输入和输出视频文件路径
    input_video_path = "Videos/TwoPeopleRunning.mp4"
    output_video_path = "Videos/VideoCartoonEffect.mp4"
    main(input_video_path, output_video_path, edge_threshold=5, color_reduction=3000)


if __name__ == "__main__":
    main()
