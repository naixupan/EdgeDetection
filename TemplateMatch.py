# 开发日期：2024年7月1日
# 文件名称：TemplateMatch.py
# 功能描述：对图片进行模板匹配
# 开发人员：何广鹏



import cv2
import matplotlib.pyplot as plt

def crop_image_using_template_matching(image_path, template_path):
    # 读取原始图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image from {image_path}")
        return

    # 读取模板图像
    template = cv2.imread(template_path)
    if template is None:
        print(f"Failed to read template from {template_path}")
        return

    # 将彩色图像转换为灰度图像
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(template.shape) == 3 and template.shape[2] == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 进行模板匹配
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    # 找到匹配结果中的最大得分的位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 获取模板的宽度和高度
    template_width = template.shape[1]
    template_height = template.shape[0]

    # 确定匹配区域
    top_left = max_loc
    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

    # 在原始图像上绘制匹配区域的矩形
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # 使用 matplotlib 显示图像
    plt.imshow(image, cmap='gray')
    plt.show()


image_path = 'F:/images/result/result_pic/cuopian/20240524_071125709_0.BMP'
template_path = './ModelImages/CropImage.bmp'


crop_image_using_template_matching(image_path, template_path)