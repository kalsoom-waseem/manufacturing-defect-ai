import os
import torch
import numpy as np
import pytest
from PIL import Image
from src.pipeline.dataset import NEUDefectDataset
from src.pipeline.transforms import get_transforms
from src.models.classifier import DefectClassifier


# On CI server there is no data folder — skip these tests gracefully
DATA_AVAILABLE = os.path.exists('data/NEU-DET/train/images')
requires_data = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason="Dataset not available on CI server"
)


# ── Dataset tests ─────────────────────────────────────────────

@requires_data
def test_dataset_loads_train():
    ds = NEUDefectDataset('data/NEU-DET', split='train')
    assert len(ds) > 0, "Training dataset is empty"
    assert len(ds.samples) == len(ds.labels)


@requires_data
def test_dataset_loads_val():
    ds = NEUDefectDataset('data/NEU-DET', split='val')
    assert len(ds) > 0, "Validation dataset is empty"


@requires_data
def test_dataset_loads_test():
    ds = NEUDefectDataset('data/NEU-DET', split='test')
    assert len(ds) > 0, "Test dataset is empty"


@requires_data
def test_dataset_no_overlap():
    val_ds  = NEUDefectDataset('data/NEU-DET', split='val')
    test_ds = NEUDefectDataset('data/NEU-DET', split='test')
    val_files  = set(val_ds.samples)
    test_files = set(test_ds.samples)
    overlap = val_files.intersection(test_files)
    assert len(overlap) == 0, \
        f"Val and test sets share {len(overlap)} images — data leakage!"


def test_dataset_correct_classes():
    assert NEUDefectDataset.CLASSES == [
        'crazing', 'inclusion', 'patches',
        'pitted_surface', 'rolled-in_scale', 'scratches'
    ]


@requires_data
def test_dataset_labels_in_range():
    ds = NEUDefectDataset('data/NEU-DET', split='train')
    for label in ds.labels:
        assert 0 <= label <= 5


@requires_data
def test_dataset_balance_ratio():
    ds = NEUDefectDataset('data/NEU-DET', split='train')
    stats = ds.get_stats()
    assert stats['balance_ratio'] > 0.8


# ── Transform tests ───────────────────────────────────────────

def test_train_transform_output_shape():
    transform = get_transforms('train')
    fake_img = Image.fromarray(
        np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    )
    tensor = transform(fake_img)
    assert tensor.shape == torch.Size([3, 224, 224])


def test_val_transform_output_shape():
    transform = get_transforms('val')
    fake_img = Image.fromarray(
        np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    )
    tensor = transform(fake_img)
    assert tensor.shape == torch.Size([3, 224, 224])


def test_val_transform_deterministic():
    transform = get_transforms('val')
    fake_img = Image.fromarray(
        np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    )
    tensor1 = transform(fake_img)
    tensor2 = transform(fake_img)
    assert torch.allclose(tensor1, tensor2)


def test_transform_pixel_range():
    transform = get_transforms('val')
    fake_img = Image.fromarray(
        np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    )
    tensor = transform(fake_img)
    assert tensor.min() >= -3.0
    assert tensor.max() <= 3.0


# ── Model tests ───────────────────────────────────────────────

def test_model_output_shape():
    model = DefectClassifier(num_classes=6, pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == torch.Size([4, 6])


def test_model_output_per_image():
    model = DefectClassifier(num_classes=6, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == torch.Size([1, 6])


def test_model_softmax_sums_to_one():
    model = DefectClassifier(num_classes=6, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = torch.softmax(model(x), dim=1)
    for i in range(2):
        total = out[i].sum().item()
        assert abs(total - 1.0) < 1e-5


def test_model_predicts_valid_class():
    model = DefectClassifier(num_classes=6, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    pred = out.argmax(dim=1).item()
    assert 0 <= pred <= 5


@requires_data
def test_dataset_returns_correct_types():
    ds = NEUDefectDataset(
        'data/NEU-DET', split='train',
        transform=get_transforms('train')
    )
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)
    assert image.shape == torch.Size([3, 224, 224])