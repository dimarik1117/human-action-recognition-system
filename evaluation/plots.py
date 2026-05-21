import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any, Tuple

RUSSIAN_CLASSES = [
    'стоять', 'сидеть', 'лежать', 'идти',
    'руки за головой', 'поднять руки', 'похлопать',
    'наклон вперёд', 'удар кулаком', 'удар ногой'
]

MODEL_COLORS = {
    'GRU': '#FF6B6B',
    'CNN+LSTM': '#4ECDC4',
    'TCN': '#45B7D1',
    'ST-GCN': '#1B3A6B',
    'baseline': '#95A5A6'
}

BAR_WIDTH = 0.2

FIG_SIZE_SMALL = (10, 6)
FIG_SIZE_MEDIUM = (12, 8)
FIG_SIZE_LARGE = (14, 10)

SAVE_DPI = 150

RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['grid.color'] = 'lightgray'

sns.set_palette("Set2")

def save_figure(fig, filename: str, subdir: str = ""):
    save_dir = os.path.join(RESULTS_DIR, subdir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"График сохранён: {save_path}")

def load_training_history(model_name: str, base_dir: str = RESULTS_DIR) -> Optional[Dict]:
    history_path = os.path.join(base_dir, f"{model_name.lower()}_training_history.json")
    
    if not os.path.exists(history_path):
        alt_path = os.path.join(CHECKPOINT_DIR, f"{model_name.lower()}_training_history.json")
        if os.path.exists(alt_path):
            history_path = alt_path
        else:
            print(f"Файл истории не найден: {history_path}")
            return None
    
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    return history


def load_all_histories(model_names: List[str] = None) -> Dict[str, Dict]:
    if model_names is None:
        model_names = ['GRU', 'CNN+LSTM', 'TCN', 'ST-GCN']
    
    histories = {}
    for model_name in model_names:
        history = load_training_history(model_name)
        if history is not None:
            histories[model_name] = history
    
    return histories

def plot_f1_comparison(
    model_names: List[str],
    f1_scores: List[List[float]],
    class_names: List[str] = RUSSIAN_CLASSES,
    title: str = "Сравнение F1-score по классам для четырёх моделей",
    figsize: Tuple[int, int] = (16, 9),
    save_path: Optional[str] = None
):
    x = np.arange(len(class_names))
    width = 0.22
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#A9C4EB', '#6B9FD2', '#3671B3', '#1B3A6B']
    edgecolors = ['#7A9BC2', '#4A7BAC', '#23568C', '#0F2445']
    
    for i, (name, scores) in enumerate(zip(model_names, f1_scores)):
        offset = (i - 1.5) * width
        bars = ax.bar(
            x + offset,
            scores,
            width,
            label=name,
            color=colors[i % len(colors)],
            edgecolor=edgecolors[i % len(edgecolors)],
            alpha=0.9,
            linewidth=0.5
        )
    
    ax.set_ylabel('F1-score', fontsize=14)
    ax.set_xlabel('Классы действий', fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=11)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks(np.arange(0.5, 1.05, 0.05))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()
    return fig


def plot_accuracy_comparison(
    model_names: List[str],
    accuracies: List[float],
    title: str = "Сравнение Accuracy четырёх моделей",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show_values: bool = True
):
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#A9C4EB', '#6B9FD2', '#3671B3', '#1B3A6B']
    
    bars = ax.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    if show_values:
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{acc:.3f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
    
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xlabel('Модель', fontsize=14)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylim(0.6, 0.9)
    ax.set_yticks(np.arange(0.6, 0.91, 0.05))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str,
    figsize: Tuple[int, int] = FIG_SIZE_SMALL,
    save_path: Optional[str] = None
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Эпоха', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Функция потерь', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3, color='lightgray')
    
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    if 'train_acc' in history and 'val_acc' in history:
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Эпоха', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title(f'{model_name} - Точность', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3, color='lightgray')
    else:
        ax2.text(0.5, 0.5, 'Нет данных об accuracy', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'{model_name} - Точность (нет данных)', fontsize=12)
    
    for spine in ax2.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    plt.suptitle(f'Кривые обучения модели {model_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()
    return fig


def plot_f1_curves_separate(histories: Dict[str, Dict], save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    model_colors = {
        'GRU': '#FF6B6B',
        'CNN+LSTM': '#4ECDC4',
        'TCN': '#45B7D1',
        'ST-GCN': '#1B3A6B'
    }
    
    for idx, (model_name, history) in enumerate(histories.items()):
        ax = axes[idx]
        
        if 'val_f1' not in history:
            print(f"Для модели {model_name} отсутствуют данные val_f1")
            continue
        
        epochs = np.arange(1, len(history['val_f1']) + 1)
        f1_values = history['val_f1']
        
        ax.plot(epochs, f1_values, color=model_colors.get(model_name, '#888888'), 
                linewidth=2.5, label=model_name)
        
        final_f1 = f1_values[-1]
        ax.scatter(len(epochs), final_f1, color=model_colors.get(model_name, '#888888'), 
                  s=100, zorder=5, edgecolor='black', linewidth=1)
        ax.annotate(f'{final_f1:.3f}', 
                   xy=(len(epochs) - 2, final_f1 + 0.02),
                   fontsize=10, fontweight='bold', ha='right')
        
        ax.axhline(y=final_f1, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Эпоха', fontsize=11)
        ax.set_ylabel('F1-score (валидация)', fontsize=11)
        ax.set_title(f'{model_name} – Кривая обучения', fontsize=13, fontweight='bold')
        ax.set_ylim(0.25, 0.95)
        ax.set_xlim(0, len(epochs) + 2)
        ax.grid(True, alpha=0.3, linestyle='--', color='lightgray')
        ax.legend(loc='lower right', fontsize=10)
        
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)
    
    plt.suptitle('Кривые обучения моделей (F1-score на валидации)', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()
    return fig


def plot_f1_curves_combined(histories: Dict[str, Dict], save_path: str = None):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    model_colors = {
        'GRU': '#FF6B6B',
        'CNN+LSTM': '#4ECDC4',
        'TCN': '#45B7D1',
        'ST-GCN': '#1B3A6B'
    }
    
    line_styles = ['-', '--', '-.', ':']
    
    for idx, (model_name, history) in enumerate(histories.items()):
        if 'val_f1' not in history:
            continue
        
        epochs = np.arange(1, len(history['val_f1']) + 1)
        f1_values = history['val_f1']
        line_style = line_styles[idx % len(line_styles)]
        
        ax.plot(epochs, f1_values, color=model_colors.get(model_name, '#888888'), 
                linewidth=2.5, label=model_name, linestyle=line_style)
        
        ax.scatter(len(epochs), f1_values[-1], color=model_colors.get(model_name, '#888888'), 
                  s=80, zorder=5, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Эпоха', fontsize=12)
    ax.set_ylabel('F1-score (валидация)', fontsize=12)
    ax.set_title('Сравнение кривых обучения всех моделей', fontsize=14)
    ax.set_ylim(0.25, 0.95)
    ax.set_xlim(0, max(len(h['val_f1']) for h in histories.values() if 'val_f1' in h) + 5)
    ax.grid(True, alpha=0.3, linestyle='--', color='lightgray')
    ax.legend(loc='lower right', fontsize=11)
    
    y_offset = 0.02
    for model_name, history in histories.items():
        if 'val_f1' not in history:
            continue
        final_val = history['val_f1'][-1]
        num_epochs = len(history['val_f1'])
        ax.annotate(f'{model_name}: {final_val:.3f}', 
                   xy=(num_epochs - 3, final_val + y_offset),
                   fontsize=9, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=1.0))
        y_offset += 0.015
    
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()
    return fig


def print_training_summary(histories: Dict[str, Dict]):
    print("\n" + "=" * 80)
    print("СВОДКА ПО ОБУЧЕНИЮ МОДЕЛЕЙ (НА ОСНОВЕ РЕАЛЬНЫХ ДАННЫХ)")
    print("=" * 80)
    print(f"{'Модель':<14} {'Эпох':<8} {'Нач. F1':<12} {'Фин. F1':<12} {'Лучший F1':<12} {'Эпоха лучшего':<14}")
    print("-" * 80)
    
    for model_name, history in histories.items():
        if 'val_f1' not in history:
            continue
        
        f1_values = history['val_f1']
        num_epochs = len(f1_values)
        start_f1 = f1_values[0]
        final_f1 = f1_values[-1]
        best_f1 = max(f1_values)
        best_epoch = np.argmax(f1_values) + 1
        
        print(f"{model_name:<14} {num_epochs:<8} {start_f1:<12.4f} {final_f1:<12.4f} {best_f1:<12.4f} {best_epoch:<14}")
    
    print("=" * 80)

def plot_class_performance(
    class_names: List[str],
    precision_per_class: List[float],
    recall_per_class: List[float],
    f1_per_class: List[float],
    model_name: str,
    figsize: Tuple[int, int] = FIG_SIZE_MEDIUM,
    save_path: Optional[str] = None
):
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, recall_per_class, width, label='Recall', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, f1_per_class, width, label='F1-score', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Классы действий', fontsize=12)
    ax.set_ylabel('Значение метрики', fontsize=12)
    ax.set_title(f'Производительность модели {model_name} по классам', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='lightgray')
    
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()
    return fig
