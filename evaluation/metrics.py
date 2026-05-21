import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

AVERAGE_TYPE = 'macro' # 'micro' или 'weighted'

ZERO_DIVISION = 0

ROUND_DIGITS = 4

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    precision_per_class: Optional[List[float]] = None
    recall_per_class: Optional[List[float]] = None
    f1_per_class: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_row(self, model_name: str) -> Dict[str, Any]:
        return {
            'model': model_name,
            'accuracy': round(self.accuracy, ROUND_DIGITS),
            'precision': round(self.precision, ROUND_DIGITS),
            'recall': round(self.recall, ROUND_DIGITS),
            'f1_score': round(self.f1_score, ROUND_DIGITS)
        }
    
    def print_summary(self, model_name: str = ""):
        prefix = f"{model_name}: " if model_name else ""
        print(f"\n{prefix}Результаты:")
        print(f"  Accuracy:  {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  Recall:    {self.recall:.4f}")
        print(f"  F1-score:  {self.f1_score:.4f}")

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def calculate_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = AVERAGE_TYPE,
    zero_division: int = ZERO_DIVISION
) -> float:
    return precision_score(y_true, y_pred, average=average, zero_division=zero_division)


def calculate_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = AVERAGE_TYPE,
    zero_division: int = ZERO_DIVISION
) -> float:
    return recall_score(y_true, y_pred, average=average, zero_division=zero_division)


def calculate_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = AVERAGE_TYPE,
    zero_division: int = ZERO_DIVISION
) -> float:
    return f1_score(y_true, y_pred, average=average, zero_division=zero_division)


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for c in range(num_classes):
        y_true_binary = (y_true == c).astype(int)
        y_pred_binary = (y_pred == c).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    return np.array(precision_per_class), np.array(recall_per_class), np.array(f1_per_class)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    average: str = AVERAGE_TYPE
) -> ClassificationMetrics:
    if num_classes is None:
        num_classes = len(np.unique(y_true))
    
    accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred, average=average)
    recall = calculate_recall(y_true, y_pred, average=average)
    f1 = calculate_f1_score(y_true, y_pred, average=average)
    
    precision_per_class, recall_per_class, f1_per_class = calculate_per_class_metrics(
        y_true, y_pred, num_classes
    )
    
    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        precision_per_class=precision_per_class.tolist(),
        recall_per_class=recall_per_class.tolist(),
        f1_per_class=f1_per_class.tolist()
    )


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    digits: int = 4
) -> str:
    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=digits,
        zero_division=ZERO_DIVISION
    )
