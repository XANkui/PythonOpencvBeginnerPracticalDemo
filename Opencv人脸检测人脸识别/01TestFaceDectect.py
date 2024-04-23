import cv2

# 调用熟悉的人脸分类器 识别特征类型
# 人脸 - haarcascade_frontalface_default.xml
# 人眼 - haarcascade_eye.xm
# 微笑 - haarcascade_smile.xml
face_detect = cv2.CascadeClassifier(
    r'E:\MyCommons\Study\python\opencv-4.8.0\data\haarcascades/haarcascade_frontalface_alt2.xml')

# 读取图片
photo = cv2.imread('Images/FourPeopleFace.jpg')

# cv2.imshow('photo', photo)
# 灰度处理
gray = cv2.cvtColor(photo, code=cv2.COLOR_BGR2GRAY)

# 检查人脸 按照1.1倍放到 周围最小像素为5
face_zone = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# 绘制矩形和圆形检测人脸

num = 0  # 统计人数
for x, y, w, h in face_zone:
    num = num + 1
    # 画矩形
    cv2.rectangle(photo, pt1=(x, y), pt2=(x + w, y + h), color=[0, 0, 255], thickness=2)
    # 画圆
    cv2.circle(photo, center=(x + w // 2, y + h // 2), radius=w // 2, color=[0, 255, 0], thickness=2)
    # 显示文字
    cv2.putText(photo, str(num), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

# 总人数
cv2.putText(photo, "{}people".format(num), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (142, 125, 52), 1)

cv2.imshow('result', photo)
cv2.waitKey(0)
# 释放资源
cv2.destroyAllWindows()
