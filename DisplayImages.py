# 开发日期：2024年7月13日
# 文件名称：DisplayImages
# 功能描述：使用matplotlib在一个窗口中展示多个图像
# 开发人员：何广鹏

import matplotlib.pyplot as plt
import numpy as np

fig = None
def display_images(images, titles):
    global fig  # 声明为全局变量
    # 检查图像和标题数量是否匹配
    if len(images) != len(titles):
        raise ValueError("图像数量和标题数量不匹配")

    num_images = len(images)
    num_rows = int(num_images / 2) + num_images % 2
    num_clos = 2

    fig, axes = plt.subplots(num_rows, num_clos, figsize=(10, 10))

    # 检查axes是否为二维数组
    if num_images == 1:
        axes = np.array([[axes]])
    elif num_images == 2:
        axes = np.array([[axes]])

    for i in range(num_images):
        row = i // num_clos
        col = i % num_clos

        axes[row, col].imshow(images[i])
        axes[row, col].set_title(titles[i])
        axes[row, col].axis('off')

    fig.canvas.mpl_connect('scroll_event', lambda event: zoom(event, axes))  # 连接滚动事件
    plt.show()

def zoom(event, axes):
    if event.button == 'up':
        for ax in axes.flatten():
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xrange = cur_xlim[1] - cur_xlim[0]
            yrange = cur_ylim[1] - cur_ylim[0]

            ax.set_xlim([cur_xlim[0] - xrange / 4, cur_xlim[1] + xrange / 4])
            ax.set_ylim([cur_ylim[0] - yrange / 4, cur_ylim[1] + yrange / 4])

    elif event.button == 'down':
        for ax in axes.flatten():
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xrange = cur_xlim[1] - cur_xlim[0]
            yrange = cur_ylim[1] - cur_ylim[0]

            ax.set_xlim([cur_xlim[0] + xrange / 4, cur_xlim[1] - xrange / 4])
            ax.set_ylim([cur_ylim[0] + yrange / 4, cur_ylim[1] - yrange / 4])

    fig.canvas.draw()  # 重新绘制图形