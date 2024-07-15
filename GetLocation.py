# 开发日期：2024年7月15日
# 文件名称：GetLocation.py
# 功能描述：对于检测到的缺陷，计算并显示缺陷的位置.
# 开发人员：何广鹏



import cv2
import numpy as np
import matplotlib.pyplot as plt

from DisplayImages import display_images


def get_clustered_region(contours, threshold_area, threshold_distance):
    # 存储小轮廓的坐标
    small_contour_coords = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < threshold_area:
            M = cv2.moments(cnt)
            if M['m00']!= 0:  # 添加条件判断
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                small_contour_coords.append((cx, cy))

    # 找到聚集的区域
    clustered_regions = []
    visited = [False] * len(small_contour_coords)

    for i, coord1 in enumerate(small_contour_coords):
        if visited[i]:
            continue

        cluster = [coord1]
        visited[i] = True

        for j, coord2 in enumerate(small_contour_coords[i + 1:], start=i + 1):
            if np.linalg.norm(np.array(coord1) - np.array(coord2)) < threshold_distance and not visited[j]:
                cluster.append(coord2)
                visited[j] = True

        clustered_regions.append(cluster)

    return clustered_regions

# 读取待检测图像和模板图片
image_path = 'F:/images/result/result_pic/cuopian/20240708_070338275_0.BMP'
template_path = './ModelImages/CropImage.bmp'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取图片并转化为灰度图
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

########################        模板匹配        ########################
# 进行模板匹配
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# 获取匹配结果中得分最高的位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 获取模板的宽度和高度
template_width = template.shape[1]
template_height = template.shape[0]

# 确定裁剪区域
top_left = max_loc
bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

# 裁剪图像
cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# 图像二值化
ret, binatied_image = cv2.threshold(cropped_image, 80, 180, cv2.THRESH_BINARY)

# 腐蚀操作
erode_image = cv2.erode(binatied_image, (5, 5))
open_image=cv2.morphologyEx(binatied_image,cv2.MORPH_OPEN,(5,5))


# 平滑滤波
blur_image = cv2.GaussianBlur(erode_image, (3, 3), 0)

edges = cv2.Canny(blur_image, 50, 150)
contours, _ = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

num_blobs = len(contours)

drewed_image = np.copy(cropped_image)
threshold_area = 100
ng_contour = []
count = 0
ok_contour = []
for contour in contours:

    area = cv2.contourArea(contour)
    print(f"当前斑点的面积: {area}")
    if area < threshold_area:
        ng_contour.append(contour)
    if area > 10000 and area < 21000:
        count += 1
        ok_contour.append(contour)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(drewed_image, (x, y), (x + w, y + h), (0,0,255), 2)



print(f"检测到的斑点: {num_blobs}")
print(f"面积正常的斑点:{len(ok_contour)}")

images = [image, cropped_image, binatied_image, erode_image,blur_image, edges, drewed_image]
titles = ["Original_image", "cropped_image", "binatied_image", "erode_image","blur_image", "edges", "drewed_image"]
display_images(images, titles)
plt.imshow(drewed_image)
plt.show()


# 设定小面积阈值和距离阈值
threshold_area = 100
threshold_distance = 200  # 可根据实际情况调整

clustered_regions = get_clustered_region(ng_contour, threshold_area, threshold_distance)

center_location=[]
print(clustered_regions)
for location in clustered_regions:
    if len(location)>1:
        coords_array = np.array(location)

        # 计算 x 坐标和 y 坐标的平均值，即为中心点坐标
        center_x = np.mean(coords_array[:, 0])
        center_y = np.mean(coords_array[:, 1])
        center_location.append((center_x,center_y))
print(f"center_location:{center_location}")