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