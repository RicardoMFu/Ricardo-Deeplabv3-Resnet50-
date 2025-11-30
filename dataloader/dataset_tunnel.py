# dataloader/dataset_tunnel.py
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class TunnelDataset(Dataset):
    def __init__(self, root, split="train", img_size=(512, 512)):
        """
        root: 数据根目录，如 'data'
        split: 'train'（目前你用这个就行）
        img_size: 统一 resize 的尺寸
        """
        self.root = Path(root)
        self.split = split
        self.img_size = img_size

        self.img_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"

        # 只收 jpg / jpeg / png，忽略 json
        exts = {".jpg", ".jpeg", ".png"}
        self.img_paths = sorted(
            [p for p in self.img_dir.iterdir() if p.suffix.lower() in exts]
        )
        assert len(self.img_paths) > 0, f"未在 {self.img_dir} 找到图像文件"

        # ImageNet 归一化
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # mask 名字形如 0001_colored_mask.png：用 0001_* 匹配
        mask_candidates = list(self.mask_dir.glob(img_path.stem + "_*.png"))
        if len(mask_candidates) == 0:
            raise FileNotFoundError(f"找不到 mask: {img_path.stem}_*.png")
        mask_path = mask_candidates[0]

        # 1) 读图
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 单通道类别编号

        # 2) resize（图像可以双线性；mask 必须 NEAREST）
        image = F.resize(image, self.img_size)
        mask = F.resize(mask, self.img_size, interpolation=Image.NEAREST)

        # 3) 转 tensor & 归一化
        image = F.to_tensor(image)
        image = F.normalize(image, self.mean, self.std)

        mask_np = np.array(mask, dtype=np.int64)  # 0/1/2/3
        mask = torch.from_numpy(mask_np)  # [H,W] long

        return image, mask
