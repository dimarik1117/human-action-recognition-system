import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_cm
from typing import List, Optional, Tuple, Dict, Any
import os

RUSSIAN_CLASSES = [
    'стоять', 'сидеть', 'лежать', 'идти',
    'руки за головой', 'поднять руки', 'похлопать',
    'наклон вперёд', 'удар кулаком', 'удар ногой'
]

ENGLISH_CLASSES = [
    'stand', 'sit', 'lie', 'walk',
    'hands_behind_head', 'raise_hands', 'clap',
    'bend_forward', 'punch', 'kick'
]

CMAP = 'Blues'

DEFAULT_FIGSIZE = (12, 10)

SAVE_DPI = 150

RESULTS_DIR = "results"

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = False
) -> np.ndarray:
    cm = sklearn_cm(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
    
    return cm


def get_confusion_matrix_normalized(cm: np.ndarray) -> np.ndarray:
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return np.nan_to_num(cm_norm)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str] = RUSSIAN_CLASSES,
    title: str = "Матрица ошибок",
    normalize: bool = True,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    cmap: str = CMAP,
    save_path: Optional[str] = None,
    show_values: bool = True,
    fmt: str = '.1f'
):
    plt.figure(figsize=figsize)
    
    if normalize:
        display_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        display_cm = np.nan_to_num(display_cm) * 100
        fmt = '.0f'
        cbar_label = 'Процент (%)'
    else:
        display_cm = cm
        fmt = 'd'
        cbar_label = 'Количество'
    
    sns.heatmap(
        display_cm,
        annot=show_values,
        fmt=fmt,
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': cbar_label},
        vmin=0,
        vmax=100 if normalize else None
    )
    
    plt.title(f'Матрица ошибок модели {title}', fontsize=16, pad=20)
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight')
        print(f"Матрица ошибок сохранена: {save_path}")
    
    plt.show()


def plot_confusion_matrices_comparison(
    matrices: Dict[str, np.ndarray],
    classes: List[str] = RUSSIAN_CLASSES,
    save_dir: Optional[str] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE
):
    n_models = len(matrices)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (model_name, cm) in enumerate(matrices.items()):
        ax = axes[idx]
        
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm) * 100
        
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.1f',
            cmap=CMAP,
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
            cbar_kws={'label': '%'},
            vmin=0,
            vmax=100
        )
        
        ax.set_title(f"{model_name}", fontsize=12)
        ax.set_xlabel('Предсказанный класс', fontsize=10)
        ax.set_ylabel('Истинный класс', fontsize=10)
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(classes, rotation=0, fontsize=8)
    
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Сравнение матриц ошибок', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'confusion_matrices_comparison.png')
        plt.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight')
        print(f"Сравнение матриц ошибок сохранено: {save_path}")
    
    plt.show()


def print_confusion_matrix_analysis(
    cm: np.ndarray,
    classes: List[str] = RUSSIAN_CLASSES,
    top_k: int = 5
):
    print("\nАнализ матрицы ошибок:")
    print("=" * 60)
    
    errors = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                errors.append({
                    'true': classes[i],
                    'pred': classes[j],
                    'count': cm[i, j]
                })
    
    errors.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"Наиболее частые ошибки (топ {min(top_k, len(errors))}):")
    for i, err in enumerate(errors[:top_k]):
        print(f"  {i+1}. {err['true']} → {err['pred']}: {err['count']} раз(а)")
    
    print("\nТочность по каждому классу (диагональ):")
    for i, class_name in enumerate(classes):
        total = cm[i, :].sum()
        correct = cm[i, i]
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {class_name}: {acc:.1f}% ({correct}/{total})")
    
    print("=" * 60)
