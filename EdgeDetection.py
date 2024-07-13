# 开发日期：2024年7月12日
# 文件名称：EdgeDetection.py
# 功能描述：对花篮图片进行边缘检测，获取各区域的面积，并统计数量。其他需求后续处理
# 开发人员：何广鹏
# 更新日期：2024年7月13日
# 更新内容：修改了斑点检测的检测策略和显示的方法

import cv2
import numpy as np
import matplotlib.pyplot as plt

from DisplayImages import display_images

# 读取待检测图像和模板图片
image_path = 'F:/images/result/result_pic/cuopian/20240524_071125709_0.BMP'
template_path = './ModelImages/Crop_Model_Mid.bmp'

image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)     #读取图片并转化为灰度图
template = cv2.imread(template_path,cv2.IMREAD_GRAYSCALE)

########################        模板匹配        ########################
# 进行模板匹配
result = cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)

# 获取匹配结果中得分最高的位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 获取模板的宽度和高度
template_width = template.shape[1]
template_height = template.shape[0]

# 确定裁剪区域
top_left = max_loc
bottom_right = (top_left[0] + template_width,top_left[1] + template_height)

# 裁剪图像
cropped_image = image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]

# 图像二值化
ret,binatied_image=cv2.threshold(cropped_image, 80, 180, cv2.THRESH_BINARY);

# 腐蚀操作
erode_image = cv2.erode(binatied_image,(5,5))
plt.imshow(erode_image, cmap='gray')
plt.show()

# 平滑滤波
blur_image = cv2.GaussianBlur(erode_image,(3,3),0)

edges = cv2.Canny(blur_image, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

num_blobs = len(contours)

drewed_image=np.copy(cropped_image)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(drewed_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    area = cv2.contourArea(contour)
    print(f"当前斑点的面积: {area}")


#显示效果
# plt.imshow(cropped_image, cmap='gray')
# plt.show()
print(f"Number of detected blobs: {num_blobs}")

images=[image,cropped_image,binatied_image,erode_image,blur_image,edges,drewed_image]
titles=["Original_image","cropped_image","binatied_image","erode_image","blur_image","edges","drewed_image"]
display_images(images,titles)