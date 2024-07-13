# 开发日期：2024年7月13日
# 文件名称：CropImage.py
# 功能描述：使用模板匹配确定目标区域并裁剪图片
# 开发人员：何广鹏

import cv2
import matplotlib.pyplot as plt

def crop_image_using_template_matching(image_path, template_path):
    # 读取原始图像和模板图像
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)

    # 将彩色图像转换为灰度图像。（请注意，如果不转换位灰度图像的话，可能会报错）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 进行模板匹配
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    # 找到匹配结果中的最大得分的位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 获取模板的宽度和高度
    template_width = template.shape[1]
    template_height = template.shape[0]

    # 确定裁剪区域
    top_left = max_loc
    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

    # 裁剪图像
    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # 显示裁剪后的图像
    # 使用 matplotlib 显示图像
    plt.imshow(cropped_image, cmap='gray')
    plt.show()

# 替换为您的图像和模板的路径
image_path = 'F:/images/result/result_pic/cuopian/20240524_071125709_0.BMP'
template_path = './ModelImages/CropImage.bmp'

crop_image_using_template_matching(image_path, template_path)