# 开发日期：2024年7月12日
# 文件名称：EdgeDetection.py
# 功能描述：对花篮图片进行边缘检测，获取各区域的面积，并统计数量。其他需求后续处理
# 开发人员：何广鹏

import cv2
import numpy as np
import matplotlib


# 读取图片；将图像转化为灰度图

image_dir='F:/images/result/result_pic/cuopian/20240524_025123220_0.BMP'
image = cv2.imread(image_dir,cv2.IMREAD_GRAYSCALE)   # 使用OpenCV读取图像，并以灰度模式加载图像
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(image)

# 裁剪

cropped_image=image[577:860, 923:10712]

# cv2.imshow("corpped_image", cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 图像二值化

ret,binatied_image=cv2.threshold(cropped_image, 80, 180, cv2.THRESH_BINARY);

# cv2.imshow("binatied_image", binatied_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 平滑滤波
blur_image = cv2.GaussianBlur(binatied_image,(3,3),0)

cv2.imshow("blur_image", blur_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# 边缘检测/斑点检测

edge_image = cv2.Canny(blur_image,100,200)

cv2.imshow("edge_image", edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 查找轮廓
contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 统计轮廓数量即图形数量
count = len(contours)

print(count)

for contour in contours:
    area = cv2.contourArea(contour)
    print("area:",area)


# 读取二值图像
# binary_image = cv2.imread('binary_image.jpg', cv2.IMREAD_GRAYSCALE)

# 标记连通区域
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binatied_image)

# 统计图形数量
count = num_labels - 1  # 减去背景

print(count)