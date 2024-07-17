# 开发日期：2024年7月16日
# 文件名称：RotateImage.py
# 功能描述：对图片进行旋转操作，角度自定
# 开发人员：何广鹏

import cv2
import os
import matplotlib.pyplot as plt
from math import fabs,sin,cos,radians

'''
函数名称：rotate_image
输入：
@image：待旋转的图片
@angle：旋转角度，其中，正数为逆时针旋转，负数为顺时针旋转
'''
def rotate_image(image,angle,save=False,savedir=None,imageName=None):

    height,width=image.shape[:2]
    scale = 1.0
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    new_H = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    new_W = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
    M[0, 2] += (new_W - width) / 2
    M[1, 2] += (new_H - height) / 2
    rotate = cv2.warpAffine(image, M, (new_W, new_H), borderValue=(0, 0, 0))
    print(rotate.shape)
    plt.imshow(rotate, cmap='gray')
    plt.show()
    if save==True:
        cv2.imwrite(os.path.join(savedir, imageName),rotate)
        print(f"{imageName}旋转并保存！")



image_path="F:/images/mdl_image"
save_path="F:/images/result/result_pic/rotated_image"
imagelist = os.listdir(image_path)

for imageName in imagelist:


    image=cv2.imread(os.path.join(image_path, imageName))
    rotate_image(image,-90,save=True,savedir=save_path,imageName=imageName)