import pytesseract
from PIL import Image
path="Images/ChineseWord.jpg"
image=Image.open(path)
text=pytesseract.image_to_string(image,lang='chi_sim')
print(text)#打印输出识别文字
