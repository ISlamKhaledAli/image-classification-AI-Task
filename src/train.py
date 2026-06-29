"""
train.py — Training & evaluation loops + curve plotting.
"""
import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct      += (outputs.argmax(1) == labels).sum().item()
        total        += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        correct      += (outputs.argmax(1) == labels).sum().item()
        total        += imgs.size(0)
    return running_loss / total, correct / total


def run_phase(model, train_loader, val_loader, criterion, optimizer,
              scheduler, device, epochs, history, ckpt_path, best_val_acc):
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        tag = " <- best" if vl_acc > best_val_acc else ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), ckpt_path)

        print(f"  Epoch [{epoch:02d}/{epochs}]  "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f}  |  "
              f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f}{tag}")

    return best_val_acc


def plot_curves(history, phase1_epochs, save_path="data/training_curves.png"):
    total  = len(history["train_loss"])
    epochs = range(1, total + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train", color="#E63946", lw=2)
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#457B9D", lw=2, ls="--")
    ax1.axvline(phase1_epochs + 0.5, color="gray", ls=":", label="Fine-tune")
    ax1.set_title("Loss Curve", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a*100 for a in history["train_acc"]], label="Train", color="#E63946", lw=2)
    ax2.plot(epochs, [a*100 for a in history["val_acc"]],   label="Val",   color="#457B9D", lw=2, ls="--")
    ax2.axvline(phase1_epochs + 0.5, color="gray", ls=":", label="Fine-tune")
    ax2.set_title("Accuracy Curve", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Curves saved -> {save_path}")
