import pytesseract
from PIL import Image
path="Images/ChineseWord.jpg"
image=Image.open(path)
text=pytesseract.image_to_string(image,lang='chi_sim')
print(text)#打印输出识别文字


def overlay_mask(mask, frame, face_x, face_y, face_width, face_height, offset_x=0, offset_y=0):
    # 获取面具的大小和位置
    mask_height, mask_width = mask.shape[:2]

    # 获取面具覆盖的区域
    mask_y = int(face_y - face_height / 2 + offset_y)
    mask_y_end = int(mask_y + face_height)
    mask_x = int(face_x - face_width / 2 + offset_x)
    mask_x_end = int(mask_x + face_width)

    # 调整面具大小
    resized_mask = resize_mask(mask, face_width, face_height)

    # 将面具叠加到视频帧上
    for c in range(0, 3):
        try:
            frame[mask_y:mask_y_end, mask_x:mask_x_end, c] = \
                resized_mask[:, :, c] * (resized_mask[:, :, 3] / 255.0) + \
                frame[mask_y:mask_y_end, mask_x:mask_x_end, c] * (1.0 - resized_mask[:, :, 3] / 255.0)
        except ValueError:
            pass

    return frame



import cv2

# 加载人脸和鼻子检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
nose_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_mcs_nose.xml")

# 打开视频文件
cap = cv2.VideoCapture('Videos/GirlFace.mp4')

while True:
    # 读取一帧视频
    ret, frame = cap.read()
    if not ret:
        break

    # 将视频帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 鼻子检测
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=2.5, minNeighbors=10, minSize=(20, 20))
        for (nx, ny, nw, nh) in noses:
            # 在鼻子位置上添加特效，这里仅作示例q
            cv2.rectangle(frame, (x+nx, y+ny), (x+nx+nw, y+ny+nh), (255, 0, 0), 2)

    # 显示结果
    cv2.imshow('Face and Nose Detection', frame)

    # 检测按键输入
    key = cv2.waitKey(1)
    if key == ord('q'):  # 按 'q' 键退出
        break

# 释放视频捕获对象
cap.release()
cv2.destroyAllWindows()