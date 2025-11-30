# evaluate_metrics.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from dataloader.dataset_tunnel import TunnelDataset
from models.deeplab_model import get_deeplabv3_resnet50


# ================================
# compute mIoU & mPA
# ================================
def compute_metrics(cm):
    eps = 1e-6
    num_classes = cm.shape[0]

    IoUs = []
    PAs = []

    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP

        IoU = TP / (TP + FP + FN + eps)
        PA = TP / (TP + FN + eps)

        IoUs.append(IoU)
        PAs.append(PA)

    return IoUs, PAs, np.mean(IoUs), np.mean(PAs)


def evaluate():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---------------------------
    # 1. 加载模型（安全加载）
    # ---------------------------
    model = get_deeplabv3_resnet50(num_classes=4, pretrained=False)

    model_path = "checkpoints/best_deeplabv3_resnet50.pth"

    state_dict = torch.load(model_path, map_location=device)
    new_state = {}

    # 过滤掉 aux_classifier（你训练时包含，但预测时没有）
    for k, v in state_dict.items():
        if k.startswith("aux_classifier"):
            continue
        new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.to(device).eval()

    # ---------------------------
    # 2. 加载训练集做评估
    # ---------------------------
    dataset = TunnelDataset(root="data", split="train", img_size=(512, 512))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    cm = np.zeros((4, 4), dtype=np.int64)

    # ---------------------------
    # 3. 遍历数据
    # ---------------------------
    for img, mask in loader:
        img = img.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            pred = model(img)["out"].argmax(1)[0]

        gt = mask[0]

        cm += confusion_matrix(
            gt.cpu().numpy().flatten(),
            pred.cpu().numpy().flatten(),
            labels=[0, 1, 2, 3],
        )

    # ---------------------------
    # 4. 指标计算
    # ---------------------------
    IoUs, PAs, mIoU, mPA = compute_metrics(cm)

    print("\n====== Evaluation Results ======")
    print("Class IoU:", IoUs)
    print("Class PA :", PAs)
    print(f"mIoU     : {mIoU:.4f}")
    print(f"mPA      : {mPA:.4f}")

    # ---------------------------
    # 5. 保存图像
    # ---------------------------
    os.makedirs("Evaluation_plot", exist_ok=True)

    # 混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("Evaluation_plot/confusion_matrix.png", dpi=300)
    plt.close()

    # IoU
    plt.figure(figsize=(6, 5))
    plt.bar(range(4), IoUs)
    plt.xticks(range(4), ["BG", "C1", "C2", "C3"])
    plt.title("Class IoU")
    plt.savefig("Evaluation_plot/class_iou.png", dpi=300)
    plt.close()

    # PA
    plt.figure(figsize=(6, 5))
    plt.bar(range(4), PAs, color="green")
    plt.xticks(range(4), ["BG", "C1", "C2", "C3"])
    plt.title("Class PA")
    plt.savefig("Evaluation_plot/class_pa.png", dpi=300)
    plt.close()

    print("\n📁 结果已保存至 Evaluation_plot/")


if __name__ == "__main__":
    evaluate()
