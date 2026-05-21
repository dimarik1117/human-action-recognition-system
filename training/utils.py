import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from typing import Optional, Dict, Any, Tuple, List

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MOMENTUM = 0.9

SCHEDULER_TYPE = 'reduce_on_plateau'  # 'cosine', 'step'
REDUCE_ON_PLATEAU_FACTOR = 0.5
REDUCE_ON_PLATEAU_PATIENCE = 5
REDUCE_ON_PLATEAU_MIN_LR = 1e-6
STEP_LR_STEP_SIZE = 30
STEP_LR_GAMMA = 0.1
COSINE_T_MAX = 50

EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001

RANDOM_SEED = 42

class EarlyStopping:
    def __init__(
        self,
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = EARLY_STOPPING_MIN_DELTA,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    momentum: float = DEFAULT_MOMENTUM
) -> optim.Optimizer:
    if optimizer_type.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Неизвестный тип оптимизатора: {optimizer_type}")


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = SCHEDULER_TYPE,
    **kwargs
) -> object:
    if scheduler_type == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', REDUCE_ON_PLATEAU_FACTOR),
            patience=kwargs.get('patience', REDUCE_ON_PLATEAU_PATIENCE),
            min_lr=kwargs.get('min_lr', REDUCE_ON_PLATEAU_MIN_LR)
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', COSINE_T_MAX)
        )
    elif scheduler_type == 'step':
        return StepLR(
            optimizer,
            step_size=kwargs.get('step_size', STEP_LR_STEP_SIZE),
            gamma=kwargs.get('gamma', STEP_LR_GAMMA)
        )
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Неизвестный тип планировщика: {scheduler_type}")

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_metric: float,
    filepath: str,
    additional_info: Optional[Dict] = None
):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metric': val_metric
    }
    if additional_info:
        checkpoint.update(additional_info)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Чекпоинт сохранён: {filepath}")


def load_checkpoint(
    model: nn.Module,
    filepath: str,
    optimizer: Optional[optim.Optimizer] = None,
    device: str = 'cpu'
) -> Tuple[int, float]:
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    val_metric = checkpoint.get('val_metric', 0.0)
    
    print(f"Чекпоинт загружен: {filepath} (эпоха {epoch}, метрика {val_metric:.4f})")
    return epoch, val_metric

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed установлен: {seed}")


def get_device() -> str:
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Используется MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Используется CPU")
    return device

def save_training_history(history: Dict[str, List[float]], model_name: str, save_dir: str = "results"):
    import json
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    history_converted = convert(history)
    
    save_path = os.path.join(save_dir, f"{model_name}_training_history.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(history_converted, f, indent=4, ensure_ascii=False)
    
    print(f"История обучения сохранена: {save_path}")
