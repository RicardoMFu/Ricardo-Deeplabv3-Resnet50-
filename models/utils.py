# utils.py
import torch


def pixel_accuracy(pred, mask):
    """
    简单像素准确率计算
    pred: [B,H,W]
    mask: [B,H,W]
    """
    correct = (pred == mask).float().sum()
    total = mask.numel()
    return (correct / total).item()
