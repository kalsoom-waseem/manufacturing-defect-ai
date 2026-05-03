import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import copy
from src.pipeline.dataset import NEUDefectDataset
from src.pipeline.transforms import get_transforms
from src.models.classifier import DefectClassifier


def train(config):

    # ── 1. Device ──────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # ── 2. Save config ─────────────────────────────────────────
    os.makedirs(config['experiment_dir'], exist_ok=True)
    with open(os.path.join(config['experiment_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config['experiment_dir']}/config.json")

    # ── 3. Datasets ────────────────────────────────────────────
    train_dataset = NEUDefectDataset(
        config['data_dir'], split='train',
        transform=get_transforms('train')
    )
    val_dataset = NEUDefectDataset(
        config['data_dir'], split='val',
        transform=get_transforms('val')
    )
    test_dataset = NEUDefectDataset(
        config['data_dir'], split='test',
        transform=get_transforms('test')
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")

    # ── 4. DataLoaders ─────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )

    # ── 5. Model ───────────────────────────────────────────────
    model = DefectClassifier(num_classes=6, pretrained=True).to(device)
    print(f"\nTrainable parameters: {model.count_trainable_params():,}")

    # ── 6. Loss function ───────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── 7. Optimiser ───────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=1e-4
    )

    # ── 8. Learning rate scheduler ─────────────────────────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ── 9. Training loop ───────────────────────────────────────
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': []}

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>9}")
    print("-" * 55)

    for epoch in range(config['epochs']):

        # ── TRAIN PHASE ──
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total

        # ── VAL PHASE ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # ── LOG ──
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"{epoch+1:>6} {avg_train_loss:>12.4f} "
              f"{train_acc*100:>9.2f}% "
              f"{avg_val_loss:>10.4f} "
              f"{val_acc*100:>8.2f}%")

        # ── SCHEDULER ──
        scheduler.step(avg_val_loss)

        # ── EARLY STOPPING ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            os.makedirs('models', exist_ok=True)
            torch.save(best_model_state, 'models/best_model.pth')
            print(f"  ✅ Best model saved (val_loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break

    # ── 10. Final test evaluation ──────────────────────────────
    print("\n" + "="*55)
    print("FINAL TEST EVALUATION")
    print("="*55)

    model.load_state_dict(best_model_state)
    model.eval()

    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # ── 11. Save history ───────────────────────────────────────
    with open(os.path.join(config['experiment_dir'],
                           'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. History saved.")
    return model, history


if __name__ == "__main__":
    config = {
        'data_dir': 'data/NEU-DET',
        'epochs': 15,
        'batch_size': 32,
        'learning_rate': 0.001,
        'early_stopping_patience': 5,
        'experiment_dir': 'experiments/run_001'
    }
    train(config)