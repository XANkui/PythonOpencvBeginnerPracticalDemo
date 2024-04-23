"""
视频图像处理基础操作
    视频捕获/存储/提取/合成/合并
"""

import cv2
import os


def vedioCaptureWrite(saveName):
    """
    视频捕捉与保存
    :param saveName: 类似 output.avi
    :return:
    """
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # 创建 VideoWriter 对象，用于保存视频
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # 循环读取摄像头帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 在视频上绘制一个简单的示例文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Recording...', (10, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # 写入帧到输出视频
        out.write(frame)

        # 显示帧
        cv2.imshow('frame', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def videoExtraction(videoName):
    """
    提取视频中的某一帧保存为图片
    :param videoName: 类似 TwoPeopleRunning.mp4
    :return:
    """
    # 打开视频文件
    cap = cv2.VideoCapture(videoName)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Unable to open video.")
        exit()

    # 选择要提取的帧索引
    frame_index = 100  # 例如，提取第 100 帧

    # 设置视频帧索引
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # 读取帧
    ret, frame = cap.read()

    # 检查帧是否成功读取
    if not ret:
        print("Error: Unable to read frame.")
        exit()

    # 保存帧为图像文件
    cv2.imwrite('Images/videoExtraction.jpg', frame)

    # 关闭视频文件
    cap.release()

    print(f"Frame {frame_index} extracted and saved as output_frame.jpg")


def imagesCompositingVideo(imageFolder, frame):
    """
    图片合成图片
    :param imageFolder: 图片文件夹，例如 'Images'
    :param frame: 每秒多少帧，例如 30帧每秒，1帧每秒
    :return:
    """

    # 图片所在文件夹
    images_folder = imageFolder

    # 读取文件夹中的所有图片文件名
    image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

    # 指定输出视频文件名
    output_video = 'Videos/imagesCompositingVideo.mp4'

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 获取第一张图片的尺寸
    first_image = cv2.imread(os.path.join(images_folder, image_files[0]))
    height, width, _ = first_image.shape

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_video, fourcc, frame, (width, height))

    # 逐张读取图片并写入视频
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        img = cv2.imread(image_path)
        out.write(img)

    # 释放资源
    out.release()


def mergeVideo(videoName1, videoName2):
    """
    合并视频
    :param videoName1: 视频名称，类似 'Videos/CatRun.mp4'
    :param videoName2: 视频名称，类似 'Videos/TwoPeopleRunning.mp4'
    :return:
    """
    # 打开第一个视频文件
    cap1 = cv2.VideoCapture(videoName1)

    # 打开第二个视频文件
    cap2 = cv2.VideoCapture(videoName2)

    # 获取视频信息
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter('Videos/mergeVideo.mp4', fourcc, fps, (width, height))

    # 逐帧读取并写入第一个视频文件
    while cap1.isOpened():
        ret, frame = cap1.read()
        if not ret:
            break
        out.write(frame)

    # 逐帧读取并写入第二个视频文件
    while cap2.isOpened():
        ret, frame = cap2.read()
        if not ret:
            break
        out.write(frame)

    # 释放资源
    cap1.release()
    cap2.release()
    out.release()


def test():
    # 添加测试函数
    vedioCaptureWrite('output.avi')


if __name__ == "__main__":
    test()

