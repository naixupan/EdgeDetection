# 开发日期：2024年7月12日
# 文件名称：EdgeDetection.py
# 功能描述：对花篮图片进行边缘检测，获取各区域的面积，并统计数量。其他需求后续处理
# 开发人员：何广鹏
# 更新日期：2024年7月13日
# 更新内容：修改了斑点检测的检测策略和显示的方法
# 更新日期：2024年7月16日
# 更新内容：图像的模版匹配和裁剪使用封装好的函数


import cv2
import numpy as np
import matplotlib.pyplot as plt

from DisplayImages import display_images
from CropImage import cropimage

# 读取待检测图像和模板图片
image_path = 'F:/images/result/result_pic/cuopian/20240524_071125709_0.BMP'
template_path = './ModelImages/Model_Image_All.bmp'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取图片并转化为灰度图
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)



cropped_image = cropimage(image,template)

# 图像二值化
ret, binatied_image = cv2.threshold(cropped_image, 80, 180, cv2.THRESH_BINARY)

# 腐蚀操作
erode_image = cv2.erode(binatied_image, (5, 5))
dilate_image=cv2.dilate(erode_image,kernel=(5,5),iterations=16)
plt.imshow(erode_image, cmap='gray')
plt.show()

# 平滑滤波
blur_image = cv2.GaussianBlur(dilate_image, (3, 3), 0)

edges = cv2.Canny(blur_image, 35, 75)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

num_blobs = len(contours)

drewed_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
threshold_area = 100
ng_contour=[]
ok_count=0


for contour in contours:


    area = cv2.contourArea(contour)
    if area>14000 and area<23000:
        ok_count+=1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(drewed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(f"当前斑点的面积: {area}")
    if area<threshold_area:
        ng_contour.append(contour)


# 显示效果
# plt.imshow(cropped_image, cmap='gray')
# plt.show()
print(f"Number of detected blobs: {num_blobs}")

images = [image, cropped_image, binatied_image, erode_image,dilate_image, blur_image, edges, drewed_image]
titles = ["Original_image", "cropped_image", "binatied_image", "erode_image", "dilate_image","blur_image", "edges", "drewed_image"]
display_images(images, titles)
plt.imshow(drewed_image,cmap="gray")
plt.show()
print(ok_count)
