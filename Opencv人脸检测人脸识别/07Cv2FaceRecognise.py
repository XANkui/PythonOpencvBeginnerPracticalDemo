"""
简单进行人脸训练与识别
    1、准备训练数据集：
    2、提取局部纹理特征：
    3、构建特征向量：
    4、训练模型：
    5、人脸识别：
"""

import cv2
import os
import numpy as np


def load_training_data(data_dir):
    """
    从指定目录加载训练数据集
    :param data_dir:(str) 包含训练图像的目录路径
    :return:tuple 包含训练图像和对应标签的元组 (faces, labels)
    """

    if not isinstance(data_dir, str) or not data_dir.strip():
        raise ValueError("Invalid data directory path.")

    faces = []  # 存储人脸图像
    labels = []  # 存储人脸标签

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(root, file)
                label = os.path.basename(root)
                face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if face_img is not None:
                    faces.append(face_img)
                    labels.append(int(label))

    if not faces or not labels:
        raise ValueError("No valid training data found in the directory:", data_dir)

    return faces, labels


def preprocess_images(faces):
    """
    对图像列表进行预处理，调整图像大小为100x100像素
    :param faces: (list) 包含人脸图像的列表
    :return: list 预处理后的人脸图像列表
    """
    if not isinstance(faces, list) or not faces:
        raise ValueError("Invalid input: faces must be a non-empty list of images.")

    preprocessed_faces = []
    for face in faces:
        if face is not None:
            face = cv2.resize(face, (100, 100))  # 调整图像大小
            preprocessed_faces.append(face)
    return preprocessed_faces


def train_lbph(faces, labels):
    """
    使用 LBPH 算法训练人脸识别器
    :param faces: (list) 包含训练图像的列表
    :param labels: (list) 包含训练图像对应标签的列表
    :return: cv2.face_LBPHFaceRecognizer: 训练完成的 LBPH 人脸识别器模型
    """
    if not isinstance(faces, list) or not faces:
        raise ValueError("Invalid input: faces must be a non-empty list of images.")

    if not isinstance(labels, list) or not labels:
        raise ValueError("Invalid input: labels must be a non-empty list of integers.")

    if len(faces) != len(labels):
        raise ValueError("Mismatch in the number of faces and labels.")

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(labels))
    return model


def load_test_image(image_path):
    """
    加载测试图像
    :param image_path: (str) 测试图像文件路径
    :return: numpy.ndarray 加载的测试图像
    """
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError("Invalid image path.")

    test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        raise ValueError("Failed to load test image from path:", image_path)
    return test_image


def preprocess_test_image(test_image):
    """
    预处理测试图像，调整大小为100x100像素
    :param test_image: (numpy.ndarray) 待处理的测试图像
    :return: numpy.ndarray 预处理后的测试图像
    """

    if test_image is None:
        raise ValueError("Invalid input: test_image is None.")

    test_image = cv2.resize(test_image, (100, 100))  # 调整图像大小
    return test_image


def recognize_face(model, test_image):
    """
    使用训练好的模型识别人脸
    :param model: (cv2.face_LBPHFaceRecognizer) 训练完成的 LBPH 人脸识别器模型
    :param test_image: (numpy.ndarray) 待识别的测试图像
    :return: tuple 识别结果的标签和置信度 (label, confidence)
    """
    if model is None or not isinstance(model, cv2.face_LBPHFaceRecognizer):
        raise ValueError("Invalid model: model must be a trained LBPH face recognizer.")

    if test_image is None:
        raise ValueError("Invalid input: test_image is None.")

    label, confidence = model.predict(test_image)
    return label, confidence


def test_face_recognition(data_dir, test_image_path):
    """
    测试人脸识别器
    :param data_dir: (str) 包含训练图像的目录路径
    :param test_image_path: (str) 测试图像文件路径
    :return: tuple 识别结果的标签和置信度 (label, confidence)
    """
    # 加载训练数据集
    faces, labels = load_training_data(data_dir)

    # 预处理训练数据集
    preprocessed_faces = preprocess_images(faces)

    # 训练 LBPH 人脸识别器
    model = train_lbph(preprocessed_faces, labels)

    # 读取测试图像
    test_image = load_test_image(test_image_path)

    # 预处理测试图像
    preprocessed_test_image = preprocess_test_image(test_image)

    # 进行人脸识别
    label, confidence = recognize_face(model, preprocessed_test_image)

    return label, confidence


# 测试人脸识别器
if __name__ == "__main__":
    data_dir = "Images/Face/Train"
    test_image_path = "Images/Face/Test/Test_Peter.png"
    label, confidence = test_face_recognition(data_dir, test_image_path)
    print("Predicted label:", label)
    print("Confidence:", confidence)
