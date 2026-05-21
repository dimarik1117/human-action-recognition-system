import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gru_model import GRUModel
from models.cnn_lstm_model import CNNLSTMModel
from models.tcn_model import TCNModel
from models.stgcn_model import STGCNModel
from training.utils import get_device, load_checkpoint

BATCH_SIZE = 32
RESULTS_DIR = "results"
CLASSES = [
    'stand', 'sit', 'lie', 'walk',
    'hands_behind_head', 'raise_hands', 'clap',
    'bend_forward', 'punch', 'kick'
]
RUSSIAN_CLASSES = [
    'стоять', 'сидеть', 'лежать', 'идти',
    'руки за головой', 'поднять руки', 'похлопать',
    'наклон вперёд', 'удар кулаком', 'удар ногой'
]

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    model_type: str = 'gru'
) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            landmarks = batch['landmarks'].to(device)
            labels = batch['label'].to(device)
            
            if model_type == 'stgcn':
                batch_size, seq_len, feat_dim = landmarks.shape
                landmarks = landmarks.view(batch_size, seq_len, 33, 2)
            
            outputs = model(landmarks)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'classification_report': classification_report(
            all_labels, all_preds,
            target_names=CLASSES,
            zero_division=0
        )
    }
    
    print(f"\nРезультаты оценки модели:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    
    return results


def load_and_evaluate(
    model_class,
    checkpoint_path: str,
    test_loader: DataLoader,
    device: str,
    model_type: str,
    **model_kwargs
) -> Dict[str, Any]:
    model = model_class(**model_kwargs)
    model = model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Модель загружена из {checkpoint_path} (эпоха {checkpoint.get('epoch', '?')})")
    
    return evaluate_model(model, test_loader, device, model_type)

def evaluate_all_models(
    test_loader: DataLoader,
    checkpoint_dir: str = "checkpoints",
    num_classes: int = 10
) -> Dict[str, Dict[str, Any]]:
    device = get_device()
    results = {}
    
    models_config = {
        'gru': {
            'class': GRUModel,
            'kwargs': {
                'input_size': 66,
                'hidden_size': 128,
                'num_layers': 2,
                'num_classes': num_classes,
                'dropout': 0.3,
                'bidirectional': False
            },
            'checkpoint': os.path.join(checkpoint_dir, 'gru_best.pth'),
            'type': 'gru'
        },
        'cnn_lstm': {
            'class': CNNLSTMModel,
            'kwargs': {
                'num_joints': 33,
                'num_coords': 2,
                'cnn_hidden_dims': [64, 128],
                'cnn_kernel_sizes': [3, 3],
                'lstm_hidden_size': 128,
                'lstm_num_layers': 2,
                'lstm_dropout': 0.3,
                'lstm_bidirectional': False,
                'fc_hidden_size': 64,
                'num_classes': num_classes,
                'dropout': 0.3
            },
            'checkpoint': os.path.join(checkpoint_dir, 'cnn_lstm_best.pth'),
            'type': 'cnn_lstm'
        },
        'tcn': {
            'class': TCNModel,
            'kwargs': {
                'input_size': 66,
                'num_channels': [64, 128, 256],
                'kernel_size': 5,
                'num_blocks': 4,
                'num_classes': num_classes,
                'fc_hidden_size': 128,
                'dropout': 0.3
            },
            'checkpoint': os.path.join(checkpoint_dir, 'tcn_best.pth'),
            'type': 'tcn'
        },
        'stgcn': {
            'class': STGCNModel,
            'kwargs': {
                'in_channels': 2,
                'num_classes': num_classes,
                'hidden_channels': [64, 128, 256],
                'num_joints': 33,
                'temporal_kernel_size': 9,
                'dropout': 0.3,
                'partition_strategy': 'spatial'
            },
            'checkpoint': os.path.join(checkpoint_dir, 'stgcn_best.pth'),
            'type': 'stgcn'
        }
    }
    
    for model_name, config in models_config.items():
        print("\n" + "=" * 60)
        print(f"Оценка модели {model_name.upper()}")
        print("=" * 60)
        
        if os.path.exists(config['checkpoint']):
            try:
                model = config['class'](**config['kwargs'])
                model = model.to(device)
                
                checkpoint = torch.load(config['checkpoint'], map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Модель загружена из {config['checkpoint']}")
                
                eval_results = evaluate_model(model, test_loader, device, config['type'])
                results[model_name] = eval_results
            except Exception as e:
                print(f"Ошибка при загрузке/оценке модели {model_name}: {e}")
                results[model_name] = None
        else:
            print(f"Чекпоинт не найден: {config['checkpoint']}")
            results[model_name] = None
    
    return results

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str] = RUSSIAN_CLASSES,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    plt.figure(figsize=figsize)
    
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Процент (%)'}
    )
    
    plt.title(title, fontsize=14)
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Матрица ошибок сохранена: {save_path}")
    
    plt.show()


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    print("\n" + "=" * 80)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(f"{'Модель':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-score':<12}")
    print("-" * 80)
    
    for model_name, res in results.items():
        if res is not None:
            print(f"{model_name.upper():<15} {res['accuracy']:<12.4f} {res['precision']:<12.4f} {res['recall']:<12.4f} {res['f1_score']:<12.4f}")
        else:
            print(f"{model_name.upper():<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    print("=" * 80)


def print_classification_report(results: Dict[str, Any], model_name: str):
    if results is not None:
        print(f"\nClassification Report для {model_name.upper()}:")
        print(results['classification_report'])
