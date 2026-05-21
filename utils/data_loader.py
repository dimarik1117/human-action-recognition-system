import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import random

WINDOW_SIZE = 30
STRIDE = 15
NUM_JOINTS = 33
FEATURES_PER_FRAME = 66

USE_AUGMENTATION = True

BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True

DATA_ROOT = "data/processed"
ANNOTATIONS_ROOT = "data/annotations"

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

CLASSES = [
    'stand',              # стоять
    'sit',                # сидеть
    'lie',                # лежать
    'walk',               # идти
    'hands_behind_head',  # руки за головой
    'raise_hands',        # поднять руки
    'clap',               # похлопать
    'bend_forward',       # наклон вперёд
    'punch',              # удар кулаком
    'kick'                # удар ногой
]

NUM_CLASSES = len(CLASSES)

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

class SkeletonDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        annotations_file: str,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        use_augmentation: bool = False,
        augmenter=None,
        transform=None
    ):
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.stride = stride
        self.use_augmentation = use_augmentation
        self.augmenter = augmenter
        self.transform = transform
        
        self.samples = self._load_annotations(annotations_file)
        
        print(f"Загружено {len(self.samples)} образцов из {annotations_file}")
    
    def _load_annotations(self, annotations_file: str) -> List[Dict]:
        import csv
        
        samples = []
        with open(annotations_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    'skeleton_path': row['skeleton_path'],
                    'label': int(row['label'])
                })
        
        return samples
    
    def _load_skeleton(self, path: str) -> np.ndarray:
        file_path = self.data_root / path
        if not file_path.exists():
            alt_path = self.data_root / f"{path}.npy"
            if alt_path.exists():
                file_path = alt_path
            else:
                raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        data = np.load(file_path)
        
        if data.ndim == 2:
            n_frames = data.shape[0]
            data = data.reshape(n_frames, NUM_JOINTS, 2)
        elif data.ndim == 3:
            pass
        else:
            raise ValueError(f"Неожиданная размерность данных: {data.shape}")
        
        return data.astype(np.float32)
    
    def _extract_windows(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_frames = landmarks.shape[0]
        windows = []
        masks = []
        
        start = 0
        while start + self.window_size <= n_frames:
            window = landmarks[start:start + self.window_size]
            window = window.reshape(self.window_size, -1)
            windows.append(window)
            masks.append(np.ones(self.window_size))
            start += self.stride
        
        if start < n_frames:
            remaining = n_frames - start
            window = landmarks[start:n_frames]
            window = window.reshape(remaining, -1)
            
            pad_width = self.window_size - remaining
            pad = np.zeros((pad_width, FEATURES_PER_FRAME))
            window = np.vstack([window, pad])
            windows.append(window)
            
            mask = np.concatenate([np.ones(remaining), np.zeros(pad_width)])
            masks.append(mask)
        
        return np.array(windows), np.array(masks)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        landmarks = self._load_skeleton(sample['skeleton_path'])
        label = sample['label']
        
        windows, masks = self._extract_windows(landmarks)
        
        if len(windows) == 0:
            windows = np.zeros((1, self.window_size, FEATURES_PER_FRAME))
            masks = np.zeros((1, self.window_size))
        
        window_idx = 0
        if self.use_augmentation and len(windows) > 1:
            window_idx = random.randint(0, len(windows) - 1)
        
        window = windows[window_idx]
        mask = masks[window_idx]
        
        if self.use_augmentation and self.augmenter is not None:
            window_reshaped = window.reshape(self.window_size, NUM_JOINTS, 2)
            window_reshaped = self.augmenter(window_reshaped)
            window = window_reshaped.reshape(self.window_size, -1)
        
        result = {
            'landmarks': torch.from_numpy(window).float(),
            'label': torch.tensor(label, dtype=torch.long),
            'mask': torch.from_numpy(mask).float()
        }
        
        if self.transform:
            result = self.transform(result)
        
        return result

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    landmarks = torch.stack([item['landmarks'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    return {
        'landmarks': landmarks,
        'label': labels,
        'mask': masks
    }


def get_data_loaders(
    data_root: str,
    annotations_root: str,
    batch_size: int = BATCH_SIZE,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    num_workers: int = NUM_WORKERS,
    use_augmentation: bool = USE_AUGMENTATION,
    augmenter=None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    train_annotations = os.path.join(annotations_root, 'train.csv')
    val_annotations = os.path.join(annotations_root, 'val.csv')
    test_annotations = os.path.join(annotations_root, 'test.csv')
    
    train_dataset = SkeletonDataset(
        data_root=data_root,
        annotations_file=train_annotations,
        window_size=window_size,
        stride=stride,
        use_augmentation=use_augmentation,
        augmenter=augmenter
    )
    
    val_dataset = SkeletonDataset(
        data_root=data_root,
        annotations_file=val_annotations,
        window_size=window_size,
        stride=stride,
        use_augmentation=False,
        augmenter=None
    )
    
    test_dataset = SkeletonDataset(
        data_root=data_root,
        annotations_file=test_annotations,
        window_size=window_size,
        stride=stride,
        use_augmentation=False,
        augmenter=None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn
    )
    
    info = {
        'num_classes': NUM_CLASSES,
        'classes': CLASSES,
        'class_to_idx': CLASS_TO_IDX,
        'idx_to_class': IDX_TO_CLASS,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset)
    }
    
    print(f"\nДатасет загружен:")
    print(f"  Train: {len(train_dataset)} образцов, {len(train_loader)} батчей")
    print(f"  Val: {len(val_dataset)} образцов, {len(val_loader)} батчей")
    print(f"  Test: {len(test_dataset)} образцов, {len(test_loader)} батчей")
    print(f"  Классов: {NUM_CLASSES}")
    
    return train_loader, val_loader, test_loader, info

def create_sequences(
    landmarks: np.ndarray,
    labels: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE
) -> Tuple[np.ndarray, np.ndarray]:
    n_frames = landmarks.shape[0]
    flat_landmarks = landmarks.reshape(n_frames, -1)
    
    X = []
    y = []
    
    start = 0
    while start + window_size <= n_frames:
        window = flat_landmarks[start:start + window_size]
        window_labels = labels[start:start + window_size]
        
        from collections import Counter
        label = Counter(window_labels).most_common(1)[0][0]
        
        X.append(window)
        y.append(label)
        start += stride
    
    if start < n_frames:
        remaining = n_frames - start
        window = flat_landmarks[start:n_frames]
        window_labels = labels[start:n_frames]
        
        pad_width = window_size - remaining
        pad = np.zeros((pad_width, FEATURES_PER_FRAME))
        window = np.vstack([window, pad])
        
        label = Counter(window_labels).most_common(1)[0][0] if len(window_labels) > 0 else 0
        
        X.append(window)
        y.append(label)
    
    return np.array(X), np.array(y)
