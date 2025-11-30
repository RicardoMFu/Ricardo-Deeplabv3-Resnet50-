from PIL import Image
import numpy as np

mask = np.array(Image.open("data/train/masks/0023_colored_mask.png"))
print(np.unique(mask))

mask = np.array(Image.open("data/train/masks/0023_colored_mask.png"))
print(mask.shape)
print(np.unique(mask))
print(mask)  # 查看局部内容
