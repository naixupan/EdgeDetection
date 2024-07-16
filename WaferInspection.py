# 开发日期：2024年7月16日
# 文件名称：WaferInspection.py
# 功能描述：检测花篮中硅片的位置，并标记
# 开发人员：何广鹏

import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime

from CropImage import cropimage


# 读取图片
image_path = 'F:/images/result/result_pic/cuopian/20240527_100845399_0.BMP'
template_path = './ModelImages/Model_Image_All.bmp'
image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
template=cv2.imread(template_path,cv2.IMREAD_GRAYSCALE)

print("原始图像，模版图像读取完成！")
cropped_image=cropimage(image,template)
print("模版匹配完成！")

# 筛选灰度值
'''
硅片在图片中的灰度值的范围为:45-75,宽度在3个像素值左右，长为186左右
'''
lower_gray=45
upper_gray=58

# 筛选灰度值在范围内的像素
print("开始筛选像素")
mask = cv2.inRange(cropped_image, lower_gray, upper_gray)
# kernel = np.array([[0, 23, 0],
#                    [0, 23, 0],
#                    [0, 23, 0]], dtype=np.uint8)
# erode_image=cv2.erode(mask,kernel=kernel)

kernel = np.array([[0, 15, 0],
                   [0, 15, 0],
                   [0, 15, 0]], dtype=np.uint8)
opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
dilate_mask=cv2.dilate(opened_mask,kernel=kernel,iterations=32)

# 平滑滤波
blur_image = cv2.GaussianBlur(dilate_mask, (5, 5), 0)
edges = cv2.Canny(blur_image, 50, 150)

# print(mask.shape)
# for i in range(mask.shape[0]):
#     raw_count=0
#     for j in range(mask.shape[1]):
#         if mask[i,j]>0:
#             raw_count+=1
#
#         if raw_count>3:
#             mask[i,j-5:j]=0
#             raw_count=0

print("筛选完毕！！")
plt.imshow(blur_image,cmap='gray')
plt.show()

# 获取满足要求的区域的轮廓
contours, _ = cv2.findContours(blur_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"轮廓数量：{len(contours)}")
# 根据轮廓绘制直线
output_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
count=0
for contour in contours:
    area=cv2.contourArea(contour)
    if area>320 :
        x, y, w, h = cv2.boundingRect(contour)
        if h>115:
            print(f"高度为：{h}")
            cv2.line(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count+=1

plt.imshow(output_image,cmap='gray')
plt.show()
print(f"检测到的线条数量为：{count}")

time=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
cv2.imwrite(f"ResultImage/Wafer-{time}.jpg",output_image)



