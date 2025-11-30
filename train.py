# train.py

import os
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataloader.dataset_tunnel import TunnelDataset
from models.deeplab_model import get_deeplabv3_resnet50


# ============================
# 工具：计算像素准确率
# ============================
def pixel_accuracy(pred, mask):
    correct = (pred == mask).float().sum()
    total = mask.numel()
    return (correct / total).item()


# ============================
# 冻结与解冻 backbone
# ============================
def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    print(">>> Backbone 已冻结（freeze）")


def unfreeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = True
    print(">>> Backbone 已解冻（unfreeze）")


# ============================
# 数据加载
# ============================
def get_dataloaders(root="data", img_size=(512, 512), batch_size=2, val_ratio=0.2):

    full_dataset = TunnelDataset(root=root, split="train", img_size=img_size)
    n_total = len(full_dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    print(f"总样本: {n_total}, 训练: {n_train}, 验证: {n_val}")
    return train_loader, val_loader


# ============================
# 训练一个 epoch
# ============================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, print_freq=50):
    model.train()
    running_loss = 0.0

    for step, (images, masks) in enumerate(loader, 1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)["out"]

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # 控制打印频率
        if step % print_freq == 0:
            print(
                f"  [Epoch {epoch}] step {step}/{len(loader)} | "
                f"batch loss: {loss.item():.4f}"
            )

    return running_loss / len(loader.dataset)


# ============================
# 验证
# ============================
@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        acc = pixel_accuracy(preds, masks)
        total_acc += acc
        n_batches += 1

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_acc / n_batches

    print(f"  [Val Epoch {epoch}] loss: {avg_loss:.4f}, pixel acc: {avg_acc:.4f}")
    return avg_loss, avg_acc


# ============================
# 主训练程序
# ============================
def main():
    # ==== 超参数 ====
    root = "data"
    img_size = (512, 512)
    num_classes = 4
    batch_size = 2
    num_epochs = 60
    lr = 1e-4
    weight_decay = 1e-4
    val_ratio = 0.2

    # ==== GPU ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ==== 数据 ====
    train_loader, val_loader = get_dataloaders(
        root=root, img_size=img_size, batch_size=batch_size, val_ratio=val_ratio
    )

    # ==== 模型 ====
    model = get_deeplabv3_resnet50(num_classes=num_classes, pretrained=True).to(device)

    # ==== 损失与优化 ====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ==== 日志文件 ====
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "train_log.csv"

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "val_acc", "best", "freeze"]
        )

    # ==== 训练 ====
    best_val_loss = float("inf")
    best_epoch = -1

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, num_epochs + 1):

        # 冻结策略
        if epoch <= 10:
            freeze_backbone(model)
            freeze_flag = "freeze"
        else:
            unfreeze_backbone(model)
            freeze_flag = "unfreeze"

        print(f"\n===== Epoch {epoch}/{num_epochs} =====")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, print_freq=50
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

        # 打印
        print(
            f"Epoch {epoch} | train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | val pixel acc: {val_acc:.4f}"
        )

        # 判断是否最佳
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = save_dir / "best_deeplabv3_resnet50.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  >>> 保存最佳模型到 {best_path}")

        # 写入 CSV
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, train_loss, val_loss, val_acc, int(is_best), freeze_flag]
            )

    print(f"\n==========================")
    print(f"训练完成！最优模型出现在第 **{best_epoch} 轮**")
    print(f"==========================\n")


if __name__ == "__main__":
    main()
