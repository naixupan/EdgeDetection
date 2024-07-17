# 开发日期：2024年7月17日
# 文件名称：WaferDetect.py
# 功能描述：使用模版匹配来检测花篮中的硅片
# 开发人员：何广鹏

import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime


time1 = datetime.datetime.now()


image_path="F:/images/result/result_pic/quepian/20240526_221807837_0.BMP"
template_mid="ModelImages/Model_Wafer.bmp"
template_top="ModelImages/Model_Wafer_Top.bmp"
template_botm="ModelImages/Model_Wafer_Bottom.bmp"

image=cv2.imread(image_path,cv2.COLOR_BGR2GRAY)
template_top=cv2.imread(template_top,cv2.COLOR_BGR2GRAY)
template_mid=cv2.imread(template_mid,cv2.COLOR_BGR2GRAY)
template_botm=cv2.imread(template_botm,cv2.COLOR_BGR2GRAY)

if len(image.shape) == 3 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
if len(template_top.shape) == 3 and template_top.shape[2] == 3:
    template = cv2.cvtColor(template_top, cv2.COLOR_BGR2GRAY)
if len(template_mid.shape) == 3 and template_mid.shape[2] == 3:
    template = cv2.cvtColor(template_mid, cv2.COLOR_BGR2GRAY)
if len(template_botm.shape) == 3 and template_botm.shape[2] == 3:
    template = cv2.cvtColor(template_botm, cv2.COLOR_BGR2GRAY)

# 进行模版匹配
result_top=cv2.matchTemplate(image,template_top,cv2.TM_CCOEFF_NORMED)
result_mid=cv2.matchTemplate(image,template_mid,cv2.TM_CCOEFF_NORMED)
result_botm=cv2.matchTemplate(image,template_botm,cv2.TM_CCOEFF_NORMED)

# 设置模版匹配阈值
threshold =0.8

# 找到匹配度大于阈值的位置
loc_top = np.where(result_top >= threshold)
loc_mid = np.where(result_mid >= threshold)
loc_botm = np.where(result_botm >= threshold)
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
count=0
for i, pt in enumerate(zip(*loc_top[::-1])):
        cv2.rectangle(output_image, pt, (pt[0] + template_top.shape[1], pt[1] + template_top.shape[0]),(0,255,0), 2)
        count+=1
for i, pt in enumerate(zip(*loc_mid[::-1])):
        cv2.rectangle(output_image, pt, (pt[0] + template_mid.shape[1], pt[1] + template_mid.shape[0]),(0,255,0), 2)
        # count += 1
for i, pt in enumerate(zip(*loc_botm[::-1])):
        cv2.rectangle(output_image, pt, (pt[0] + template_botm.shape[1], pt[1] + template_botm.shape[0]),(0,255,0), 2)
        # count += 1

time2 = datetime.datetime.now()
plt.imshow(output_image,cmap="gray")
plt.show()

print(f"用时：{time2-time1}")