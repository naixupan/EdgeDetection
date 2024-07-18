# 开发日期：2024年7月18日
# 文件名称：RegionManager.py
# 功能描述：在模版匹配中对重复区域的判定
# 开发人员：何广鹏

class RegionManager:
    def __init__(self):
        self.drawn_regions = []

    # 判断区域是否重复
    def is_region_duplicated(self, new_x, new_y, new_width, new_height, threshold=0.5):
        for x, y, width, height in self.drawn_regions:
            intersection_area = max(0, min(x + width, new_x + new_width) - max(x, new_x)) * max(0, min(y + height, new_y + new_height) - max(y, new_y))
            total_area = width * height + new_width * new_height - intersection_area
            if intersection_area / total_area > threshold:
                return True
        return False

    # 将已绘制区域添加到列表
    def add_drawn_region(self, x, y, width, height):
        self.drawn_regions.append((x, y, width, height))
