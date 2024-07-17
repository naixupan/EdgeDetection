# 开发日期：2024年7月16日
# 文件名称：WaferInspection.py
# 功能描述：检测花篮中硅片的位置，并标记
# 开发人员：何广鹏

# 更新日期：2024年7月17日
# 更新内容：在筛选灰度之前添加一个平滑操作，提高了检测硅片的稳定性
#          增加了联通区域分析

import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime

from CropImage import cropimage

time1 = datetime.datetime.now()
save = 0
# 读取图片
image_path = 'F:/images/result/result_pic/suipian/20240707_085532150_0.BMP'
template_path = './ModelImages/Model_Image_All.bmp'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
print("原始图像，模版图像读取完成！")
cropped_image = cropimage(image, template)
print("模版匹配完成！")
blur_image_crop = cv2.GaussianBlur(cropped_image, (5, 5), 0)
# 筛选灰度值
'''
硅片在图片中的灰度值的范围为:45-75,宽度在3个像素值左右，长为186左右
'''
# 经过平滑操作后的灰度阈值
lower_gray = 70

upper_gray = 90

# 筛选灰度值在范围内的像素
print("开始筛选像素")
mask = cv2.inRange(blur_image_crop, lower_gray, upper_gray)

# kernel = np.array([[0, 23, 0],
#                    [0, 23, 0],
#                    [0, 23, 0]], dtype=np.uint8)
# erode_image=cv2.erode(mask,kernel=kernel)

kernel = np.array([[0, 15, 0],
                   [0, 15, 0],
                   [0, 15, 0]], dtype=np.uint8)
erode_image=cv2.erode(mask,kernel=kernel)
opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
dilate_mask = cv2.dilate(opened_mask, kernel=kernel, iterations=16)
plt.imshow(dilate_mask, cmap='gray')
plt.show()


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


# 获取满足要求的区域的轮廓
contours, _ = cv2.findContours(blur_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"轮廓数量：{len(contours)}")
# 根据轮廓绘制直线
output_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
count = 0
maxarea = 0
wafer_rectangle=[]
for contour in contours:
    area = cv2.contourArea(contour)
    if area > maxarea:
        maxarea = area
    # print(f"面积为：{area}")
    if area > 1000 and area < 10000:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 180:
            print(f"高度为：{h}")
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.line(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1
            wafer_rectangle.append([x,y,w,h])

# 在输出图片上添加相关信息
font = cv2.FONT_HERSHEY_SIMPLEX
text = f"detected lines:{count}"
# 添加文字
cv2.putText(output_image, text, (102, 342), font, 5, color=(0, 255, 255), thickness=4)
plt.imshow(output_image, cmap='gray')
plt.show()
cv2.imshow("output_image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"检测到的线条数量为：{count}")
print(output_image.shape)

if save == 1:
    time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    cv2.imwrite(f"ResultImage/Wafer-{time}.jpg", output_image)

'''
考虑根据初步的检测结果再进一步进行灰度的划分
1、首先将表示硅片范围的坐标进行保存，保存形式为列表
2、对每个小区域进行灰度阈值的划分，划分的图片为cropped_image
'''
wafer_rectangle=sorted(wafer_rectangle,key=lambda coord:(coord[0],-coord[1]),reverse=False)
print(wafer_rectangle)
for i in wafer_rectangle:
    print(i)

# 对于检测到的区域重新进行灰度的筛选
# for rectangle in wafer_rectangle:
#     x,y,w,h=rectangle[0],rectangle[1],rectangle[2],rectangle[3]
#     region=blur_image_crop[y:y + h, x:x + w]
#     mask_region = cv2.inRange(region, lower_gray, upper_gray)
#     opened_mask_region = cv2.morphologyEx(mask_region, cv2.MORPH_OPEN, kernel)
    # plt.imshow(opened_mask_region, cmap='gray')
    # plt.show()

##################      进行联通区域分析        #################
