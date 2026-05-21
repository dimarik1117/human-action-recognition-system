import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import random
import torch
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.extract_landmarks import (
    extract_landmarks_from_video,
    save_landmarks_to_file,
    TARGET_FPS,
    CONFIDENCE_THRESHOLD
)
from preprocessing.normalize import normalize_skeleton
from preprocessing.interpolate import interpolate_missing_points
from preprocessing.filter import exponential_smoothing, SMOOTHING_ALPHA
from preprocessing.slicer import sliding_window_slice, WINDOW_SIZE, STRIDE

from utils.data_loader import (
    SkeletonDataset,
    get_data_loaders,
    CLASSES,
    NUM_CLASSES,
    BATCH_SIZE,
    WINDOW_SIZE as DL_WINDOW_SIZE,
    STRIDE as DL_STRIDE
)

from training.train_gru import train_gru
from training.train_cnn_lstm import train_cnn_lstm
from training.train_tcn import train_tcn
from training.train_stgcn import train_stgcn
from training.utils import set_seed, get_device

from evaluation.compare_models import ModelComparator
from evaluation.metrics import calculate_all_metrics
from evaluation.confusion_matrix import compute_confusion_matrix

DATA_ROOT = Path("data")
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")

PROCESS_FPS = TARGET_FPS               # 30 FPS
CONF_THRESHOLD = CONFIDENCE_THRESHOLD  # 0.5
SMOOTHING_ALPHA = SMOOTHING_ALPHA      # 0.3
NORM_WINDOW_SIZE = WINDOW_SIZE         # 30
NORM_STRIDE = STRIDE                   # 15

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

TRAIN_MODELS = ['gru', 'cnn_lstm', 'tcn', 'stgcn']
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

SKIP_EXTRACTION = False
SKIP_PREPROCESSING = False
SKIP_TRAINING = False
SKIP_EVALUATION = False

def get_video_files(data_dir: Path) -> Dict[str, List[Path]]:
    video_files = {}
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / f"class_{class_idx+1:02d}_{class_name}"
        if class_dir.exists():
            videos = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mov"))
            video_files[class_name] = videos
            print(f"Класс {class_name}: {len(videos)} видео")
        else:
            print(f"Предупреждение: директория {class_dir} не найдена")
            video_files[class_name] = []
    
    return video_files


def process_all_videos(
    video_files: Dict[str, List[Path]],
    output_dir: Path,
    force_reprocess: bool = False
) -> Dict[str, List[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    skeleton_files = {}
    
    for class_name, videos in video_files.items():
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        skeleton_files[class_name] = []
        
        for video_path in videos:
            stem = video_path.stem
            output_path = class_output_dir / stem
            
            if not force_reprocess and (output_path.with_suffix('.npy')).exists():
                print(f"  Пропуск (уже обработан): {video_path.name}")
                skeleton_files[class_name].append(str(class_name / stem))
                continue
            
            print(f"  Обработка: {video_path.name}")
            
            try:
                landmarks, confidences, metadata = extract_landmarks_from_video(
                    str(video_path),
                    target_fps=PROCESS_FPS,
                    confidence_threshold=CONF_THRESHOLD
                )
                
                landmarks_flat = landmarks.reshape(landmarks.shape[0], -1)
                
                np.save(str(output_path) + '.npy', landmarks_flat)
                with open(str(output_path) + '_metadata.json', 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4, ensure_ascii=False)
                
                skeleton_files[class_name].append(str(class_name / stem))
                print(f"    Сохранено: {landmarks_flat.shape}")
                
            except Exception as e:
                print(f"    Ошибка при обработке {video_path.name}: {e}")
    
    return skeleton_files


def preprocess_skeleton_files(
    skeleton_files: Dict[str, List[str]],
    input_dir: Path,
    output_dir: Path,
    force_reprocess: bool = False
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name, files in skeleton_files.items():
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        for file_rel_path in files:
            input_path = input_dir / (file_rel_path + '.npy')
            output_path = class_output_dir / Path(file_rel_path).name
            
            if not force_reprocess and (output_path.with_suffix('.npy')).exists():
                print(f"  Пропуск (уже обработан): {file_rel_path}")
                continue
            
            print(f"  Предобработка: {file_rel_path}")
            
            try:
                landmarks_flat = np.load(input_path)
                n_frames = landmarks_flat.shape[0]
                landmarks = landmarks_flat.reshape(n_frames, 33, 2)
                
                landmarks = interpolate_missing_points(landmarks)
                
                landmarks = normalize_skeleton(landmarks)
                
                landmarks = exponential_smoothing(landmarks, alpha=SMOOTHING_ALPHA)
                
                windows = sliding_window_slice(
                    landmarks,
                    window_size=NORM_WINDOW_SIZE,
                    stride=NORM_STRIDE
                )
                
                np.save(str(output_path) + '_windows.npy', windows)
                
                landmarks_flat_out = landmarks.reshape(n_frames, -1)
                np.save(str(output_path) + '.npy', landmarks_flat_out)
                
                print(f"    Сохранено: окна {windows.shape}")
                
            except Exception as e:
                print(f"    Ошибка при предобработке {file_rel_path}: {e}")


def create_annotations(
    skeleton_files: Dict[str, List[str]],
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
    
    for class_name, files in skeleton_files.items():
        label = class_to_idx[class_name]
        for file_rel_path in files:
            all_samples.append({
                'skeleton_path': file_rel_path,
                'label': label,
                'class_name': class_name
            })
    
    print(f"\nВсего образцов: {len(all_samples)}")
    
    y = [s['label'] for s in all_samples]
    
    train_idx, temp_idx = train_test_split(
        range(len(all_samples)),
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=y,
        random_state=42
    )
    
    temp_labels = [all_samples[i]['label'] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        range(len(temp_idx)),
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_labels,
        random_state=42
    )
    
    val_idx = [temp_idx[i] for i in val_idx]
    test_idx = [temp_idx[i] for i in test_idx]
    
    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    test_samples = [all_samples[i] for i in test_idx]
    
    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    test_df = pd.DataFrame(test_samples)
    
    train_df[['skeleton_path', 'label']].to_csv(output_dir / 'train.csv', index=False)
    val_df[['skeleton_path', 'label']].to_csv(output_dir / 'val.csv', index=False)
    test_df[['skeleton_path', 'label']].to_csv(output_dir / 'test.csv', index=False)
    
    print(f"\nРазделение данных:")
    print(f"  Train: {len(train_df)} образцов")
    print(f"  Val: {len(val_df)} образцов")
    print(f"  Test: {len(test_df)} образцов")
    
    print(f"\nРаспределение по классам в Train:")
    for class_name in CLASSES:
        count = train_df[train_df['class_name'] == class_name].shape[0]
        print(f"  {class_name}: {count}")

def train_and_save_model(
    model_name: str,
    train_loader,
    val_loader,
    num_classes: int,
    checkpoint_dir: Path
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Обучение модели {model_name.upper()}")
    print(f"{'='*60}")
    
    if model_name == 'gru':
        from training.train_gru import train_gru as train_func
        history = train_func(
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            save_checkpoints=True,
            checkpoint_dir=str(checkpoint_dir),
            model_name=model_name
        )
    elif model_name == 'cnn_lstm':
        from training.train_cnn_lstm import train_cnn_lstm as train_func
        history = train_func(
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            save_checkpoints=True,
            checkpoint_dir=str(checkpoint_dir),
            model_name=model_name
        )
    elif model_name == 'tcn':
        from training.train_tcn import train_tcn as train_func
        history = train_func(
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            save_checkpoints=True,
            checkpoint_dir=str(checkpoint_dir),
            model_name=model_name
        )
    elif model_name == 'stgcn':
        from training.train_stgcn import train_stgcn as train_func
        history = train_func(
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            save_checkpoints=True,
            checkpoint_dir=str(checkpoint_dir),
            model_name=model_name
        )
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")
    
    return history


def evaluate_trained_models(
    test_loader,
    checkpoint_dir: Path,
    num_classes: int
) -> ModelComparator:
    device = get_device()
    
    from models.gru_model import GRUModel
    from models.cnn_lstm_model import CNNLSTMModel
    from models.tcn_model import TCNModel
    from models.stgcn_model import STGCNModel
    
    models_config = {
        'GRU': {
            'class': GRUModel,
            'checkpoint': checkpoint_dir / 'gru_best.pth',
            'kwargs': {
                'input_size': 66,
                'hidden_size': 128,
                'num_layers': 2,
                'num_classes': num_classes,
                'dropout': 0.3,
                'bidirectional': False
            },
            'type': 'gru'
        },
        'CNN+LSTM': {
            'class': CNNLSTMModel,
            'checkpoint': checkpoint_dir / 'cnn_lstm_best.pth',
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
            'type': 'cnn_lstm'
        },
        'TCN': {
            'class': TCNModel,
            'checkpoint': checkpoint_dir / 'tcn_best.pth',
            'kwargs': {
                'input_size': 66,
                'num_channels': [64, 128, 256],
                'kernel_size': 5,
                'num_blocks': 4,
                'num_classes': num_classes,
                'fc_hidden_size': 128,
                'dropout': 0.3
            },
            'type': 'tcn'
        },
        'ST-GCN': {
            'class': STGCNModel,
            'checkpoint': checkpoint_dir / 'stgcn_best.pth',
            'kwargs': {
                'in_channels': 2,
                'num_classes': num_classes,
                'hidden_channels': [64, 128, 256],
                'num_joints': 33,
                'temporal_kernel_size': 9,
                'dropout': 0.3,
                'partition_strategy': 'spatial'
            },
            'type': 'stgcn'
        }
    }
    
    model_results = {}
    all_labels = None
    
    for model_name, config in models_config.items():
        checkpoint_path = config['checkpoint']
        
        if not checkpoint_path.exists():
            print(f"Чекпоинт не найден: {checkpoint_path}")
            continue
        
        print(f"\nОценка модели {model_name}...")
        
        try:
            model = config['class'](**config['kwargs'])
            model = model.to(device)
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    landmarks = batch['landmarks'].to(device)
                    labels = batch['label'].to(device)
                    
                    if config['type'] == 'stgcn':
                        batch_size, seq_len, feat_dim = landmarks.shape
                        landmarks = landmarks.view(batch_size, seq_len, 33, 2)
                    
                    outputs = model(landmarks)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            model_results[model_name] = (all_labels, all_preds)
            
        except Exception as e:
            print(f"Ошибка при оценке модели {model_name}: {e}")
    
    if not model_results:
        raise RuntimeError("Не удалось оценить ни одну модель")
    
    comparator = ModelComparator()
    for name, (y_true, y_pred) in model_results.items():
        comparator.add_model(name, y_true, y_pred)
    
    return comparator

def main(args):
    print("=" * 80)
    print("СИСТЕМА РАСПОЗНАВАНИЯ ДЕЙСТВИЙ ЧЕЛОВЕКА НА ОСНОВЕ СКЕЛЕТНЫХ ДАННЫХ")
    print("=" * 80)
    
    set_seed(42)
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SKIP_EXTRACTION:
        print("\n" + "=" * 60)
        print("ШАГ 1: Извлечение скелетных данных из видео")
        print("=" * 60)
        
        video_files = get_video_files(RAW_DATA_DIR)
        
        skeleton_files = process_all_videos(
            video_files,
            PROCESSED_DATA_DIR,
            force_reprocess=args.force_reprocess
        )
        
        with open(PROCESSED_DATA_DIR / 'skeleton_files.json', 'w', encoding='utf-8') as f:
            json.dump(skeleton_files, f, indent=4, ensure_ascii=False)
    else:
        print("\nПропуск ШАГА 1 (извлечение скелетных данных)")
        with open(PROCESSED_DATA_DIR / 'skeleton_files.json', 'r', encoding='utf-8') as f:
            skeleton_files = json.load(f)

    if not SKIP_PREPROCESSING:
        print("\n" + "=" * 60)
        print("ШАГ 2: Предобработка скелетных данных")
        print("  - Интерполяция пропусков")
        print("  - Нормализация координат")
        print("  - Экспоненциальное сглаживание")
        print("  - Слайсинг на окна")
        print("=" * 60)
        
        preprocess_skeleton_files(
            skeleton_files,
            PROCESSED_DATA_DIR,
            PROCESSED_DATA_DIR,
            force_reprocess=args.force_reprocess
        )
    else:
        print("\nПропуск ШАГА 2 (предобработка)")

    print("\n" + "=" * 60)
    print("ШАГ 3: Создание аннотаций")
    print("=" * 60)
    
    skeleton_files = {}
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = PROCESSED_DATA_DIR / class_name
        if class_dir.exists():
            files = [f.stem for f in class_dir.glob("*.npy") if not f.stem.endswith('_windows')]
            skeleton_files[class_name] = [f"{class_name}/{f}" for f in files]
    
    create_annotations(skeleton_files, ANNOTATIONS_DIR)

    print("\n" + "=" * 60)
    print("ШАГ 4: Загрузка данных в PyTorch")
    print("=" * 60)
    
    from augmentation.spatial import SpatialAugmenter
    from augmentation.temporal import TemporalAugmenter
    from augmentation.noise import NoiseAugmenter
    from augmentation.spatial import SpatialAugmenter as SpatialAug
    from augmentation.temporal import TemporalAugmenter as TemporalAug
    from augmentation.noise import NoiseAugmenter as NoiseAug
    
    class CombinedAugmenter:
        def __init__(self):
            self.spatial = SpatialAug()
            self.temporal = TemporalAug()
            self.noise = NoiseAug()
        
        def __call__(self, landmarks):
            result = landmarks.copy()
            result = self.spatial(result)
            result = self.temporal(result)
            result = self.noise(result)
            return result
    
    train_augmenter = CombinedAugmenter() if not args.no_augmentation else None
    
    train_loader, val_loader, test_loader, dataset_info = get_data_loaders(
        data_root=str(PROCESSED_DATA_DIR),
        annotations_root=str(ANNOTATIONS_DIR),
        batch_size=BATCH_SIZE,
        window_size=DL_WINDOW_SIZE,
        stride=DL_STRIDE,
        use_augmentation=not args.no_augmentation,
        augmenter=train_augmenter
    )

    if not SKIP_TRAINING:
        print("\n" + "=" * 60)
        print("ШАГ 5: Обучение моделей")
        print("=" * 60)
        
        histories = {}
        
        for model_name in TRAIN_MODELS:
            history = train_and_save_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=NUM_CLASSES,
                checkpoint_dir=CHECKPOINT_DIR
            )
            histories[model_name] = history
        
        with open(RESULTS_DIR / 'training_histories.json', 'w', encoding='utf-8') as f:
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.float32):
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(item) for item in obj]
                return obj
            
            json.dump(convert(histories), f, indent=4, ensure_ascii=False)
    else:
        print("\nПропуск ШАГА 5 (обучение)")

    if not SKIP_EVALUATION:
        print("\n" + "=" * 60)
        print("ШАГ 6: Оценка и сравнение моделей")
        print("=" * 60)
        
        comparator = evaluate_trained_models(
            test_loader=test_loader,
            checkpoint_dir=CHECKPOINT_DIR,
            num_classes=NUM_CLASSES
        )
        
        comparator.save_results(str(RESULTS_DIR))
        comparator.generate_report(str(RESULTS_DIR))
        
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ СОХРАНЕНЫ")
        print(f"  - Чекпоинты: {CHECKPOINT_DIR}")
        print(f"  - Результаты: {RESULTS_DIR}")
        print("=" * 60)
    else:
        print("\nПропуск ШАГА 6 (оценка)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Система распознавания действий человека")
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Принудительно переобработать все видео')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Отключить аугментацию данных')
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Пропустить извлечение скелетных данных')
    parser.add_argument('--skip_preprocessing', action='store_true',
                        help='Пропустить предобработку')
    parser.add_argument('--skip_training', action='store_true',
                        help='Пропустить обучение')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Пропустить оценку')
    parser.add_argument('--models', nargs='+', default=TRAIN_MODELS,
                        choices=['gru', 'cnn_lstm', 'tcn', 'stgcn'],
                        help='Модели для обучения')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Размер батча')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Скорость обучения')
    
    args = parser.parse_args()
    
    TRAIN_MODELS = args.models
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    
    SKIP_EXTRACTION = args.skip_extraction
    SKIP_PREPROCESSING = args.skip_preprocessing
    SKIP_TRAINING = args.skip_training
    SKIP_EVALUATION = args.skip_evaluation
    
    main(args)