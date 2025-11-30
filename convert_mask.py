from PIL import Image
import numpy as np
import os

# 颜色到类别编号的映射（RGB 格式）
COLOR_TO_CLASS = {
    (0, 0, 0): 0,  # 背景 - 黑
    (255, 0, 0): 1,  # 渗水 - 红
    (0, 255, 0): 2,  # 脱落 - 绿
    (0, 0, 255): 3,  # 裂缝 - 蓝
}

input_dir = "data/train/masks_visualizations"  # 你的彩色 mask 目录
output_dir = "data/train/masks"  # 输出单通道类别 mask
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(input_dir, filename)).convert("RGB")
        img_np = np.array(img)

        H, W, _ = img_np.shape
        mask = np.zeros((H, W), dtype=np.uint8)

        # 遍历映射表，找到所有对应颜色的像素
        for rgb, class_id in COLOR_TO_CLASS.items():
            r, g, b = rgb
            mask[
                (img_np[:, :, 0] == r) & (img_np[:, :, 1] == g) & (img_np[:, :, 2] == b)
            ] = class_id

        # 保存单通道类别 mask
        Image.fromarray(mask).save(os.path.join(output_dir, filename))

print("✓ 转换完成，请用 np.unique 检查类别编号。")
