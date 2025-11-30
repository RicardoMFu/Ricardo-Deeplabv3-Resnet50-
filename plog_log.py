# plot_log.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================
# 读取日志 CSV
# ==========================
df = pd.read_csv("logs/train_log.csv")

# ==========================
# 保存路径：Evaluation_plot
# ==========================
save_dir = Path("Evaluation_plot")
save_dir.mkdir(exist_ok=True)
print(f"图像将保存到: {save_dir.resolve()}")

# ==========================
# 全局美化设置（精美级别）
# ==========================
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 13
plt.rcParams["axes.edgecolor"] = "#222222"
plt.rcParams["axes.linewidth"] = 1.3
plt.rcParams["axes.titlepad"] = 12
plt.rcParams["savefig.dpi"] = 300


# ==========================
# 图1：Train Loss
# ==========================
plt.figure()
plt.plot(
    df["epoch"],
    df["train_loss"],
    marker="o",
    markersize=6,
    linewidth=2.2,
    color="#1f77b4",
)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.grid(True, linestyle="--", alpha=0.45)
plt.tight_layout()
plt.savefig(save_dir / "train_loss_curve.png")
plt.close()


# ==========================
# 图2：Validation Loss
# ==========================
plt.figure()
plt.plot(
    df["epoch"],
    df["val_loss"],
    marker="s",
    markersize=6,
    linewidth=2.2,
    color="#ff7f0e",
)
plt.title("Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.grid(True, linestyle="--", alpha=0.45)
plt.tight_layout()
plt.savefig(save_dir / "val_loss_curve.png")
plt.close()


# ==========================
# 图3：Validation Pixel Accuracy
# ==========================
plt.figure()
plt.plot(
    df["epoch"], df["val_acc"], marker="^", markersize=6, linewidth=2.2, color="#2ca02c"
)
plt.title("Validation Pixel Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Pixel Accuracy")
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.45)
plt.tight_layout()
plt.savefig(save_dir / "val_acc_curve.png")
plt.close()


# ==========================
# 图4：Backbone 冻结状态（Freeze / Unfreeze）
# ==========================
plt.figure(figsize=(10, 3))
freeze_flag = df["freeze"].apply(lambda x: 1 if x == "freeze" else 0)

plt.step(df["epoch"], freeze_flag, where="mid", linewidth=2.2, color="#9467bd")
plt.title("Backbone Freeze / Unfreeze Status")
plt.xlabel("Epoch")
plt.ylabel("Status")
plt.yticks([0, 1], ["Unfreeze", "Freeze"])
plt.grid(True, linestyle="--", alpha=0.45)
plt.tight_layout()
plt.savefig(save_dir / "freeze_status_curve.png")
plt.close()


print("🎉 All beautiful plots saved into Evaluation_plot/*.png")
