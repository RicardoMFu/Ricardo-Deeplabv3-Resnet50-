import os
import numpy as np
from PIL import Image

mask_dir = "data/train/masks"

all_values = set()

for name in os.listdir(mask_dir):
    if name.endswith(".png"):
        arr = np.array(Image.open(os.path.join(mask_dir, name)))
        all_values.update(np.unique(arr).tolist())

print("所有 mask 中的类别值：", sorted(all_values))
