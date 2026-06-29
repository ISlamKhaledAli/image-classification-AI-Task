import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - saves images without opening windows
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loaders
from model import build_model
from train import run_phase, plot_curves
from evaluate import get_predictions, plot_confusion_matrix, show_predictions

def main():
    # 1. Configuration
    data_dir = "data/PetImages"
    batch_size = 32
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(data_dir):
        print(f"Dataset not found at {data_dir}. Please run 'python download_dataset.py' first.")
        return

    # 2. Data Pipeline
    print("\n--- 1. Data Pipeline ---")
    train_loader, val_loader, test_loader, class_names = get_loaders(data_dir, batch_size=batch_size)
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # 3. Model Architecture
    print("\n--- 2. Model Architecture ---")
    model = build_model(num_classes=len(class_names), freeze_backbone=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    os.makedirs("data", exist_ok=True)
    ckpt_path = "data/best_model.pth"

    # 4. Training
    print(f"\n--- 3. Training & Evaluation (for {epochs} epochs) ---")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    run_phase(model, train_loader, val_loader, criterion, optimizer,
              scheduler, device, epochs, history, ckpt_path, best_val_acc=0.0)

    # 5. Evaluation and Plots
    print("\n--- 4. Visualizations ---")
    plot_curves(history, phase1_epochs=epochs, save_path="data/training_curves.png")
    
    # Load best model for testing
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    
    print("Evaluating on Test Set...")
    labels, preds = get_predictions(model, test_loader, device)
    plot_confusion_matrix(labels, preds, class_names, save_path="data/confusion_matrix.png")
    
    try:
        show_predictions(model, test_loader, class_names, device, save_path="data/predictions.png")
    except Exception as e:
        print(f"Skipping predictions visualization due to terminal environment: {e}")

if __name__ == "__main__":
    main()
