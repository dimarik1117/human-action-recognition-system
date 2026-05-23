import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import calculate_all_metrics, ClassificationMetrics
from evaluation.confusion_matrix import (
    plot_confusion_matrix,
    plot_confusion_matrices_comparison,
    compute_confusion_matrix
)
from evaluation.plots import (
    plot_f1_comparison,
    plot_accuracy_comparison,
    load_all_histories,
    plot_f1_curves_separate,
    plot_f1_curves_combined,
    print_training_summary,
    save_figure
)

CLASS_NAMES_RU = [
    'стоять', 'сидеть', 'лежать', 'идти',
    'руки за головой', 'поднять руки', 'похлопать',
    'наклон вперёд', 'удар кулаком', 'удар ногой'
]

CLASS_NAMES_EN = [
    'stand', 'sit', 'lie', 'walk',
    'hands_behind_head', 'raise_hands', 'clap',
    'bend_forward', 'punch', 'kick'
]

RESULTS_DIR = "results"
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

TABLE_FORMAT = 'csv'

class ModelComparator:
    def __init__(self, class_names: List[str] = CLASS_NAMES_RU):
        self.class_names = class_names
        self.results: Dict[str, ClassificationMetrics] = {}
        self.confusion_matrices: Dict[str, np.ndarray] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.labels: Optional[np.ndarray] = None
    
    def add_model(
        self,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cm: Optional[np.ndarray] = None
    ):
        metrics = calculate_all_metrics(y_true, y_pred, num_classes=len(self.class_names))
        self.results[name] = metrics
        
        self.predictions[name] = y_pred
        
        if self.labels is None:
            self.labels = y_true
        
        if cm is None:
            cm = compute_confusion_matrix(y_true, y_pred)
        self.confusion_matrices[name] = cm
        
        print(f"Добавлена модель {name}: Accuracy = {metrics.accuracy:.4f}")
    
    def get_comparison_table(self) -> pd.DataFrame:
        rows = []
        for name, metrics in self.results.items():
            rows.append({
                'Модель': name,
                'Accuracy': round(metrics.accuracy, 4),
                'Precision': round(metrics.precision, 4),
                'Recall': round(metrics.recall, 4),
                'F1-score': round(metrics.f1_score, 4)
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Accuracy', ascending=False)
        return df
    
    def get_f1_table_per_class(self) -> pd.DataFrame:
        data = {'Класс': self.class_names}
        for name, metrics in self.results.items():
            if metrics.f1_per_class:
                data[name] = [round(f1, 4) for f1 in metrics.f1_per_class]
        
        return pd.DataFrame(data)
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 80)
        
        df = self.get_comparison_table()
        print(df.to_string(index=False))
        
        print("\n" + "=" * 80)
        print("F1-SCORE ПО КЛАССАМ")
        print("=" * 80)
        
        df_f1 = self.get_f1_table_per_class()
        print(df_f1.to_string(index=False))
    
    def plot_training_curves(self, checkpoint_dir: str = "checkpoints"):
        histories = load_all_histories(model_names=list(self.results.keys()), base_dir=checkpoint_dir)
        if histories:
            print(f"\nЗагружены истории для моделей: {list(histories.keys())}")
            print_training_summary(histories)
            plot_f1_curves_separate(histories, save_path=os.path.join(FIGURES_DIR, 'training_curves_separate.png'))
            plot_f1_curves_combined(histories, save_path=os.path.join(FIGURES_DIR, 'training_curves_combined.png'))
        else:
            print("\nНе найдено файлов с историей обучения. Для построения кривых обучения сначала обучите модели.")
    
    def save_results(self, output_dir: str = RESULTS_DIR):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(TABLES_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        df = self.get_comparison_table()
        df.to_csv(os.path.join(TABLES_DIR, 'comparison_table.csv'), index=False)
        print(f"Сохранено: {os.path.join(TABLES_DIR, 'comparison_table.csv')}")
        
        df_f1 = self.get_f1_table_per_class()
        df_f1.to_csv(os.path.join(TABLES_DIR, 'f1_per_class.csv'), index=False)
        print(f"Сохранено: {os.path.join(TABLES_DIR, 'f1_per_class.csv')}")
        
        for name, cm in self.confusion_matrices.items():
            np.save(os.path.join(TABLES_DIR, f'confusion_matrix_{name}.npy'), cm)
            
            plot_confusion_matrix(
                cm,
                classes=self.class_names,
                title=f'Матрица ошибок - {name}',
                save_path=os.path.join(FIGURES_DIR, f'confusion_matrix_{name}.png')
            )
        
        if self.results:
            model_names = list(self.results.keys())
            f1_scores = [metrics.f1_per_class for metrics in self.results.values() if metrics.f1_per_class]
            
            if f1_scores and all(f1_scores):
                plot_f1_comparison(
                    model_names,
                    f1_scores,
                    class_names=self.class_names,
                    title="Сравнение F1-score по классам",
                    save_path='f1_comparison.png'
                )
            
            accuracies = [metrics.accuracy for metrics in self.results.values()]
            plot_accuracy_comparison(
                model_names,
                accuracies,
                title="Сравнение точности моделей",
                save_path='accuracy_comparison.png'
            )
        
        self.plot_training_curves()
        
        print(f"\nВсе результаты сохранены в {output_dir}")

def compare_all_models(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    class_names: List[str] = CLASS_NAMES_RU,
    save_results: bool = True,
    output_dir: str = RESULTS_DIR
) -> ModelComparator:
    comparator = ModelComparator(class_names=class_names)
    
    for name, (y_true, y_pred) in model_results.items():
        comparator.add_model(name, y_true, y_pred)
    
    comparator.print_summary()
    
    if save_results:
        comparator.save_results(output_dir)
    
    return comparator

def print_comparison_table(results: Dict[str, ClassificationMetrics]):
    print("\n" + "=" * 70)
    print(f"{'Модель':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-score':<12}")
    print("-" * 70)
    
    for name, metrics in results.items():
        print(f"{name:<15} {metrics.accuracy:<12.4f} {metrics.precision:<12.4f} {metrics.recall:<12.4f} {metrics.f1_score:<12.4f}")
    
    print("=" * 70)

def save_comparison_table(results: Dict[str, ClassificationMetrics], output_path: str):
    rows = []
    for name, metrics in results.items():
        rows.append({
            'model': name,
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Таблица сохранена: {output_path}")
