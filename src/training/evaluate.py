import torch
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


CLASS_NAMES = [
    'crazing', 'inclusion', 'patches',
    'pitted_surface', 'rolled-in_scale', 'scratches'
]


def evaluate(model, dataloader, device, save_dir='experiments'):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = probs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss   = total_loss / len(all_labels)

    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds,
                                average='macro', zero_division=0)
    recall    = recall_score(all_labels, all_preds,
                             average='macro', zero_division=0)
    f1        = f1_score(all_labels, all_preds,
                         average='macro', zero_division=0)

    precision_per = precision_score(all_labels, all_preds,
                                    average=None, zero_division=0)
    recall_per    = recall_score(all_labels, all_preds,
                                 average=None, zero_division=0)
    f1_per        = f1_score(all_labels, all_preds,
                             average=None, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Loss:              {avg_loss:.4f}")
    print(f"Accuracy:          {accuracy*100:.2f}%")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")
    print(f"F1 Score (macro):  {f1:.4f}")
    print(f"\nPer-class results:")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 50)
    for i, cls in enumerate(CLASS_NAMES):
        flag = " ⚠️" if recall_per[i] < 0.80 else ""
        print(f"{cls:<20} {precision_per[i]:>10.3f} "
              f"{recall_per[i]:>8.3f} {f1_per[i]:>8.3f}{flag}")

    print(f"\n{classification_report(all_labels, all_preds, target_names=CLASS_NAMES)}")

    metrics = {
        'timestamp': datetime.now().isoformat(),
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'per_class': {
            CLASS_NAMES[i]: {
                'precision': float(precision_per[i]),
                'recall':    float(recall_per[i]),
                'f1':        float(f1_per[i])
            }
            for i in range(len(CLASS_NAMES))
        }
    }

    metrics_path = os.path.join(save_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    _plot_confusion_matrix(cm, save_dir)

    return metrics


def _plot_confusion_matrix(cm, save_dir):

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix — Raw Counts')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].tick_params(axis='x', rotation=45)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix — Normalised')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")