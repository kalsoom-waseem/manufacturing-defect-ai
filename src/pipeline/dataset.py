import os
import random
from torch.utils.data import Dataset
from PIL import Image


class NEUDefectDataset(Dataset):

    CLASSES = [
        'crazing',
        'inclusion', 
        'patches',
        'pitted_surface',
        'rolled-in_scale',
        'scratches'
    ]

    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

    def __init__(self, root_dir, split='train', transform=None,
                 test_split=0.15, random_seed=42):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.labels = []

        if split == 'train':
            images_dir = os.path.join(root_dir, 'train', 'images')
            self.samples, self.labels = self._collect(images_dir)

        else:
            images_dir = os.path.join(root_dir, 'validation', 'images')
            all_samples, all_labels = self._collect(images_dir)

            random.seed(random_seed)
            indices = list(range(len(all_samples)))
            random.shuffle(indices)

            n_test = int(len(indices) * test_split)
            test_idx = indices[:n_test]
            val_idx = indices[n_test:]

            chosen = val_idx if split == 'val' else test_idx
            self.samples = [all_samples[i] for i in chosen]
            self.labels = [all_labels[i] for i in chosen]

        self.class_counts = {}
        for cls_name in self.CLASSES:
            idx = self.CLASS_TO_IDX[cls_name]
            self.class_counts[cls_name] = self.labels.count(idx)

    def _collect(self, images_dir):
        all_samples = []
        all_labels = []

        for cls_name in self.CLASSES:
            cls_dir = os.path.join(images_dir, cls_name)

            if not os.path.exists(cls_dir):
                print(f"Warning: folder not found: {cls_dir}")
                continue

            for img_file in sorted(os.listdir(cls_dir)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_samples.append(os.path.join(cls_dir, img_file))
                    all_labels.append(self.CLASS_TO_IDX[cls_name])

        return all_samples, all_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_stats(self):
        total = len(self.samples)
        counts = self.class_counts
        balance = min(counts.values()) / max(counts.values()) if counts else 0
        return {
            'split': self.split,
            'total_samples': total,
            'class_distribution': counts,
            'balance_ratio': round(balance, 3)
        }