"""
evaluate.py — Confusion matrix, classification report, visual predictions.
"""
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_predictions(model, loader, device):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def plot_confusion_matrix(labels, preds, class_names, save_path="data/confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="gray", ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved -> {save_path}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names))


def show_predictions(model, loader, class_names, device, n=12,
                     save_path="data/predictions.png"):
    imgs, labels = next(iter(loader))
    imgs_dev = imgs[:n].to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(imgs_dev), dim=1)
        preds = probs.argmax(1).cpu()
        confs = probs.max(1).values.cpu()

    mean = torch.tensor(MEAN).view(3,1,1)
    std  = torch.tensor(STD).view(3,1,1)
    imgs = (imgs[:n] * std + mean).clamp(0,1)

    cols, rows = 6, (n + 5) // 6
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off"); continue
        ax.imshow(imgs[i].permute(1,2,0).numpy())
        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"{class_names[preds[i]]} ({confs[i]*100:.0f}%)",
                     color=color, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Test Predictions  |  Correct (Green)  Wrong (Red)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
