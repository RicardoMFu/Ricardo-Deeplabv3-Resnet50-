# predict.py
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import os

from models.deeplab_model import get_deeplabv3_resnet50


# ============================
# 颜色映射（根据你任务的类别定义）
# ============================
COLOR_MAP = {
    0: (0, 0, 0),  # 背景 - 黑色
    1: (0, 0, 255),  # 红色（例如裂缝）
    2: (0, 255, 0),  # 绿色（例如渗水）
    3: (255, 0, 0),  # 蓝色（例如破损）
}


# ============================
# 图像变换（必须与训练保持一致）
# ============================
transform_img = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
)


def decode_mask(mask_np):
    """将 0/1/2/3 mask 转彩色 mask"""
    h, w = mask_np.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in COLOR_MAP.items():
        color_mask[mask_np == cls] = color

    return color_mask


def overlay(original, color_mask, alpha=0.5):
    """彩色 mask 覆盖到原图上"""
    original = cv2.resize(original, (512, 512))
    blended = cv2.addWeighted(original, 1 - alpha, color_mask, alpha, 0)
    return blended


def predict(image_path, model_path="checkpoints/best_deeplabv3_resnet50.pth"):
    # ============================
    # 设备
    # ============================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ============================
    # 加载模型
    # ============================
    model = get_deeplabv3_resnet50(num_classes=4, pretrained=False)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # ============================
    # 读取图像
    # ============================
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform_img(img_pil).unsqueeze(0).to(device)

    # ============================
    # 推理
    # ============================
    with torch.no_grad():
        output = model(img_tensor)["out"]  # [1,4,H,W]
        pred_mask = output.argmax(dim=1)[0]  # [H,W]

    pred_np = pred_mask.cpu().numpy()

    # ============================
    # 可视化
    # ============================
    original_np = np.array(img_pil)
    color_mask = decode_mask(pred_np)
    result = overlay(original_np, color_mask)

    # ============================
    # 保存结果
    # ============================
    os.makedirs("results", exist_ok=True)

    base = os.path.basename(image_path)
    name = os.path.splitext(base)[0]

    save_mask = f"results/{name}_mask.png"
    save_overlay = f"results/{name}_overlay.png"

    cv2.imwrite(save_mask, color_mask[:, :, ::1])
    cv2.imwrite(save_overlay, result[:, :, ::1])

    print(f"🎉 预测完成！")
    print(f"✔ 彩色 mask：{save_mask}")
    print(f"✔ 覆盖可视化：{save_overlay}")


if __name__ == "__main__":
    # 修改这里测试你自己的图片
    predict("data\\train\\images\\10005.jpg")
