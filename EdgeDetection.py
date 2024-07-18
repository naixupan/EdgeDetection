# 开发日期：2024年7月12日
# 文件名称：EdgeDetection.py
# 功能描述：对花篮图片进行边缘检测，获取各区域的面积，并统计数量。其他需求后续处理
# 开发人员：何广鹏

# 更新日期：2024年7月13日
# 更新内容：修改了斑点检测的检测策略和显示的方法
# 更新日期：2024年7月16日
# 更新内容：图像的模版匹配和裁剪使用封装好的函数
# 更新日期：2024年7月17日
# 更新内容：计算斑点的中心点坐标，并根据坐标情况判断可能存在缺陷的位置


import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime

from DisplayImages import display_images
from CropImage import cropimage

# 读取待检测图像和模板图片
image_path = 'F:/images/result/result_pic/quepian/20240530_062747103_0.BMP'
template_path = './ModelImages/Model_Image_All.bmp'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取图片并转化为灰度图
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)



cropped_image = cropimage(image,template)

# 图像二值化
ret, binatied_image = cv2.threshold(cropped_image, 80, 180, cv2.THRESH_BINARY)

# 腐蚀操作
erode_image = cv2.erode(binatied_image, (3, 3))
dilate_image=cv2.dilate(erode_image,kernel=(5,5),iterations=1)


# 平滑滤波
blur_image = cv2.GaussianBlur(erode_image, (3, 3), 0)

edges = cv2.Canny(blur_image, 80, 180)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

num_blobs = len(contours)

drewed_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
threshold_area = 100
ng_contour=[]
ok_count=0
centercoordinates=[]

for contour in contours:


    area = cv2.contourArea(contour)
    if area>10000 and area<25000:
        ok_count+=1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(drewed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center=(x+w/2,y+h/2)
        centercoordinates.append(center)
    print(f"当前斑点的面积: {area}")
    if area<threshold_area:
        ng_contour.append(contour)

# 对中心点坐标进行排序
centercoordinates=sorted(centercoordinates,key=lambda coord:(coord[0],coord[1]),reverse=False)

# 显示效果
# plt.imshow(cropped_image, cmap='gray')
# plt.show()
print(f"Number of detected blobs: {ok_count}")

# images = [image, cropped_image, binatied_image, erode_image,dilate_image, blur_image, edges, drewed_image]
# titles = ["Original_image", "cropped_image", "binatied_image", "erode_image", "dilate_image","blur_image", "edges", "drewed_image"]
# display_images(images, titles)
plt.imshow(drewed_image,cmap="gray")
plt.show()
save=1
if save == 1:
    time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    cv2.imwrite(f"ResultImage/EdgeDetection-{time}.jpg", drewed_image)

#

top_coord=[]
mid_coord=[]
bottom_coord=[]

for i in centercoordinates:
    if i[1]<250:
        top_coord.append(i)
    elif i[1]<650:
        mid_coord.append(i)
    else:
        bottom_coord.append(i)

print(f"top_coord:{len(top_coord)},mid_coord:{len(mid_coord)},bottom_coord:{len(bottom_coord)}")
print(top_coord)
print(mid_coord)
print(bottom_coord)