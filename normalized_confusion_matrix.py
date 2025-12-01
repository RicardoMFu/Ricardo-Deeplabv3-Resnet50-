import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ====================================
# 你的原始混淆矩阵（从 evaluate_metrics.py 读取的）
# ====================================
cm = np.array(
    [
        [273138344, 4904061, 453564, 16630],
        [5075596, 10885181, 63531, 0],
        [1274170, 87154, 1429541, 843],
        [403135, 2918, 1702, 59214],
    ],
    dtype=float,
)

# ====================================
# 归一化（按行归一化）
# ====================================
row_sums = cm.sum(axis=1, keepdims=True)
cm_normalized = cm / row_sums

# 转换为百分比形式
cm_percent = cm_normalized * 100

# ====================================
# 绘图风格美化
# ====================================
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid", font_scale=1.4)

ax = sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=True,
    linewidths=0.5,
    linecolor="gray",
    annot_kws={"size": 14},
)

# ====================================
# 标签设置
# ====================================
classes = ["BG", "C1", "C2", "C3"]
ax.set_xticklabels(classes, rotation=0)
ax.set_yticklabels(classes, rotation=0)

plt.title("Normalized Confusion Matrix (%)", fontsize=20, pad=20)
plt.xlabel("Predicted Label", fontsize=16)
plt.ylabel("True Label", fontsize=16)

# ====================================
# 保存图像
# ====================================
os.makedirs("Evaluation_plot", exist_ok=True)
save_path = "Evaluation_plot/normalized_confusion_matrix.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"🎉 归一化混淆矩阵已保存到：{save_path}")
