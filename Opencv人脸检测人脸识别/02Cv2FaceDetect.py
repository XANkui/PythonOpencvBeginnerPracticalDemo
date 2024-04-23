"""
简单人脸识别
    1、加载分类器模型：首先，需要加载已经训练好的人脸分类器模型。OpenCV提供了训练好的分类器文件，例如haarcascade_frontalface_default.xml，用于人脸检测。
    2、读取图像：将待检测的图像读取为OpenCV的图像对象。
    3、转换为灰度图像：由于人脸检测通常不需要颜色信息，因此将图像转换为灰度图像可以加快处理速度。
    4、人脸检测：利用detectMultiScale函数进行人脸检测。该函数会返回一个矩形列表，每个矩形表示一个检测到的人脸区域的位置和大小。
    5、绘制人脸框：遍历检测到的人脸区域，利用OpenCV提供的绘制函数在原始图像上绘制矩形框，标注出人脸位置。
    6、显示结果：将绘制了人脸框的图像显示出来，或者保存到文件中。
"""


import os
import cv2


def detect_faces(image_path, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    """
    识别图像中的人脸，并绘制人脸轮廓
    :param image_path:(str)输入图像的文件路径
    :param scaleFactor:(float)用于图像尺度补偿的比例因子
    :param minNeighbors:(int)每个候选矩形应该保留的邻近数量
    :param minSize:(tuple)人脸的最小尺寸。
    :return: numpy.ndarray 绘制了人脸轮廓的图像数据；int 检测到的人脸数量
    """
    # 检查图像文件路径是否存在
    if not os.path.isfile(image_path):
        raise FileNotFoundError("Input image file not found.")

    # 加载人脸分类器
    face_cascade = cv2.CascadeClassifier(
        r'E:\MyCommons\Study\python\opencv-4.8.0\data\haarcascades\haarcascade_frontalface_default.xml')

    # 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                          minSize=minSize)

    # 人脸标签计数
    num = 0

    # 绘制人脸轮廓
    for (x, y, w, h) in faces:
        num += 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, f'Face {num}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 返回带有绘制的人脸轮廓的图像和检测到的人脸数量
    return image, len(faces)


def main():
    # 调用函数并指定输入图像文件路径
    input_image_path = 'Images/FourPeopleFace.jpg'
    detected_image, num_faces = detect_faces(input_image_path)

    # 显示检测到的人脸数量
    print("Number of faces detected:", num_faces)

    # 显示绘制了人脸轮廓的图像
    cv2.imshow('Detected Faces', detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
