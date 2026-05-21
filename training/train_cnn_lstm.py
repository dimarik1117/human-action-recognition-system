import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_lstm_model import CNNLSTMModel
from training.utils import (
    EarlyStopping,
    get_optimizer,
    get_scheduler,
    save_checkpoint,
    AverageMeter,
    set_seed,
    get_device,
    save_training_history
)

NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
OPTIMIZER_TYPE = 'adamw'
SCHEDULER_TYPE = 'reduce_on_plateau'
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001
SAVE_CHECKPOINTS = True
CHECKPOINT_DIR = "checkpoints"
PRINT_FREQ = 10

CNN_HIDDEN_DIMS = [64, 128]
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_BIDIRECTIONAL = False
DROPOUT = 0.3

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    print_freq: int = PRINT_FREQ
) -> float:
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        landmarks = batch['landmarks'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(landmarks)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.update(loss.item(), landmarks.size(0))
        pbar.set_postfix({'loss': losses.avg})
    
    return losses.avg


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int
) -> Tuple[float, float, float]:
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}", leave=False)
        
        for batch in pbar:
            landmarks = batch['landmarks'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            
            losses.update(loss.item(), landmarks.size(0))
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import f1_score
    accuracy = 100.0 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return losses.avg, accuracy, f1

def train_cnn_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    optimizer_type: str = OPTIMIZER_TYPE,
    scheduler_type: str = SCHEDULER_TYPE,
    cnn_hidden_dims: list = None,
    lstm_hidden_size: int = LSTM_HIDDEN_SIZE,
    lstm_num_layers: int = LSTM_NUM_LAYERS,
    lstm_bidirectional: bool = LSTM_BIDIRECTIONAL,
    dropout: float = DROPOUT,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    save_checkpoints: bool = SAVE_CHECKPOINTS,
    checkpoint_dir: str = CHECKPOINT_DIR,
    model_name: str = "cnn_lstm"
) -> Dict[str, Any]:
    if cnn_hidden_dims is None:
        cnn_hidden_dims = CNN_HIDDEN_DIMS
    
    device = get_device()
    print(f"\nОбучение модели CNN+LSTM")
    print(f"Устройство: {device}")
    print(f"Количество классов: {num_classes}")
    print(f"Параметры: cnn={cnn_hidden_dims}, lstm_hidden={lstm_hidden_size}, dropout={dropout}")
    
    model = CNNLSTMModel(
        num_joints=33,
        num_coords=2,
        cnn_hidden_dims=cnn_hidden_dims,
        cnn_kernel_sizes=[3, 3],
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=dropout,
        lstm_bidirectional=lstm_bidirectional,
        fc_hidden_size=64,
        num_classes=num_classes,
        dropout=dropout
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_type, learning_rate, weight_decay)
    scheduler = get_scheduler(optimizer, scheduler_type)
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'best_epoch': 0,
        'best_acc': 0.0
    }
    
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\nНачало обучения...")
    print("=" * 60)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Эпоха {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion, device, epoch)
        
        if scheduler_type == 'reduce_on_plateau':
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"\nРезультаты эпохи {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2f}%")
        
        if val_acc > history['best_acc']:
            history['best_acc'] = val_acc
            history['best_epoch'] = epoch
            
            if save_checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
                save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
            print(f"  *** Новая лучшая модель! Accuracy: {val_acc:.2f}% ***")
        
        if early_stopping(val_acc):
            print(f"\nEarly stopping сработал на эпохе {epoch}")
            break
    
    print("\n" + "=" * 60)
    print(f"Обучение завершено!")
    print(f"Лучшая точность: {history['best_acc']:.2f}% на эпохе {history['best_epoch']}")

    save_training_history(history, model_name, checkpoint_dir)
    
    return history
