import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
import json
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.extract_landmarks import (
    _init_pose_detector,
    LANDMARK_INDICES,
    CONFIDENCE_THRESHOLD
)
from preprocessing.normalize import normalize_skeleton
from preprocessing.interpolate import interpolate_missing_points
from preprocessing.filter import exponential_smoothing, SMOOTHING_ALPHA
from preprocessing.slicer import sliding_window_slice, WINDOW_SIZE, STRIDE

from models.gru_model import GRUModel
from models.cnn_lstm_model import CNNLSTMModel
from models.tcn_model import TCNModel
from models.stgcn_model import STGCNModel

from training.utils import get_device

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

MODEL_TYPE = 'stgcn'
CHECKPOINT_DIR = "checkpoints"

WINDOW_SIZE = 30
STRIDE = 15
CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_ALPHA = 0.3

SMOOTHING_WINDOW = 5
CONFIDENCE_THRESHOLD_DISPLAY = 0.6

SHOW_SKELETON = True
SHOW_CONNECTIONS = True
SHOW_FPS = True
SHOW_LANDMARK_INDICES = False

SAVE_VIDEO = False
SAVE_VIDEO_DIR = "recordings"

CLASS_NAMES = [
    'стоять', 'сидеть', 'лежать', 'идти',
    'руки за головой', 'поднять руки', 'похлопать',
    'наклон вперёд', 'удар кулаком', 'удар ногой'
]

COLOR_SKELETON = (0, 255, 0)
COLOR_JOINTS = (0, 255, 0)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_BBOX = (0, 255, 0)

SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (0, 5), (0, 6), (6, 7), (7, 8), 

    (11, 12), (11, 23), (12, 24), (23, 24),

    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),

    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),

    (23, 25), (25, 27), (27, 29), (29, 31),

    (24, 26), (26, 28), (28, 30), (30, 32)
]

class RealTimeInference:
    def __init__(
        self,
        model_type: str = MODEL_TYPE,
        checkpoint_dir: str = CHECKPOINT_DIR,
        window_size: int = WINDOW_SIZE,
        smoothing_window: int = SMOOTHING_WINDOW,
        confidence_threshold: float = CONFIDENCE_THRESHOLD_DISPLAY,
        device: str = None
    ):
        self.model_type = model_type.lower()
        self.checkpoint_dir = checkpoint_dir
        self.window_size = window_size
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        self.frame_buffer = deque(maxlen=window_size)
        self.prediction_buffer = deque(maxlen=smoothing_window)
        
        self.current_landmarks = None
        self.current_prediction = None
        self.current_confidence = 0.0
        self.fps = 0.0
        
        self.model = self._load_model()
        self.pose_detector = _init_pose_detector()
        
        print(f"\nИнициализация RealTimeInference:")
        print(f"  Модель: {model_type.upper()}")
        print(f"  Устройство: {self.device}")
        print(f"  Размер окна: {window_size} кадров")
        print(f"  Сглаживание: {smoothing_window} кадров")
    
    def _load_model(self):
        num_classes = len(CLASS_NAMES)
        
        if self.model_type == 'gru':
            model = GRUModel(
                input_size=66,
                hidden_size=128,
                num_layers=2,
                num_classes=num_classes,
                dropout=0.3,
                bidirectional=False
            )
            checkpoint_name = "gru_best.pth"
        
        elif self.model_type == 'cnn_lstm':
            model = CNNLSTMModel(
                num_joints=33,
                num_coords=2,
                cnn_hidden_dims=[64, 128],
                cnn_kernel_sizes=[3, 3],
                lstm_hidden_size=128,
                lstm_num_layers=2,
                lstm_dropout=0.3,
                lstm_bidirectional=False,
                fc_hidden_size=64,
                num_classes=num_classes,
                dropout=0.3
            )
            checkpoint_name = "cnn_lstm_best.pth"
        
        elif self.model_type == 'tcn':
            model = TCNModel(
                input_size=66,
                num_channels=[64, 128, 256],
                kernel_size=5,
                num_blocks=4,
                num_classes=num_classes,
                fc_hidden_size=128,
                dropout=0.3
            )
            checkpoint_name = "tcn_best.pth"
        
        elif self.model_type == 'stgcn':
            model = STGCNModel(
                in_channels=2,
                num_classes=num_classes,
                hidden_channels=[64, 128, 256],
                num_joints=33,
                temporal_kernel_size=9,
                dropout=0.3,
                partition_strategy='spatial'
            )
            checkpoint_name = "stgcn_best.pth"
        
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"  Модель загружена из: {checkpoint_path}")
        
        return model
    
    def _extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose_detector.process(frame_rgb)
        
        if result.pose_landmarks:
            landmarks = []
            for i, lm in enumerate(result.pose_landmarks.landmark):
                confidence = getattr(lm, 'visibility', 1.0)
                if confidence < CONFIDENCE_THRESHOLD:
                    landmarks.append([np.nan, np.nan])
                else:
                    landmarks.append([lm.x, lm.y])
            
            return np.array(landmarks, dtype=np.float32)
        
        return None
    
    def _preprocess_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        landmarks = landmarks[np.newaxis, ...]
        
        landmarks = interpolate_missing_points(landmarks)
        
        landmarks = normalize_skeleton(landmarks)
        
        if hasattr(self, '_prev_landmarks'):
            landmarks = exponential_smoothing(landmarks, alpha=SMOOTHING_ALPHA)
        
        self._prev_landmarks = landmarks.copy()
        
        return landmarks[0]
    
    def _predict(self, window: np.ndarray) -> Tuple[int, float]:
        batch_size, seq_len, num_joints, num_coords = 1, window.shape[0], 33, 2
        x = window.reshape(1, seq_len, num_joints * num_coords)
        x = torch.from_numpy(x).float().to(self.device)
        
        with torch.no_grad():
            if self.model_type == 'stgcn':
                x = window.reshape(1, seq_len, 33, 2)
                x = torch.from_numpy(x).float().to(self.device)
                logits = self.model(x)
            else:
                logits = self.model(x)
            
            probs = F.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)
        
        return pred.item(), confidence.item()
    
    def update(self, frame: np.ndarray) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        landmarks = self._extract_landmarks(frame)
        
        if landmarks is not None:
            landmarks = self._preprocess_landmarks(landmarks)
            self.current_landmarks = landmarks
            
            self.frame_buffer.append(landmarks)
            
            if len(self.frame_buffer) == self.window_size:
                window = np.array(self.frame_buffer)
                pred_class, confidence = self._predict(window)
                
                self.prediction_buffer.append(pred_class)
                
                if len(self.prediction_buffer) == self.smoothing_window:
                    from collections import Counter
                    smoothed_pred = Counter(self.prediction_buffer).most_common(1)[0][0]
                    self.current_prediction = smoothed_pred
                    self.current_confidence = confidence
        
        if self.current_prediction is not None:
            action = CLASS_NAMES[self.current_prediction]
            return action, self.current_confidence, self.current_landmarks
        
        return None, 0.0, self.current_landmarks
    
    def draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        frame_width: int,
        frame_height: int
    ) -> np.ndarray:
        points = []
        for i, (x, y) in enumerate(landmarks):
            if not np.isnan(x) and not np.isnan(y):
                px = int(x * frame_width)
                py = int(y * frame_height)
                points.append((px, py))
                
                cv2.circle(frame, (px, py), 4, COLOR_JOINTS, -1)
                
                if SHOW_LANDMARK_INDICES:
                    cv2.putText(
                        frame, str(i), (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                    )
            else:
                points.append(None)
        
        if SHOW_CONNECTIONS:
            for connection in SKELETON_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    start = points[start_idx]
                    end = points[end_idx]
                    if start is not None and end is not None:
                        cv2.line(frame, start, end, COLOR_SKELETON, 2)
        
        return frame
    
    def draw_info(
        self,
        frame: np.ndarray,
        action: Optional[str],
        confidence: float,
        fps: float
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        
        info_panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, info_panel_height), COLOR_TEXT_BG, -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(
            frame, f"Model: {self.model_type.upper()}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2
        )
        
        if SHOW_FPS:
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2
            )
        
        cv2.putText(
            frame, f"Window: {len(self.frame_buffer)}/{self.window_size}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1
        )
        
        if action and confidence >= self.confidence_threshold:
            if confidence > 0.8:
                color = (0, 255, 0)
            elif confidence > 0.6:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            
            action_text = f"Action: {action}"
            confidence_text = f" ({confidence:.1%})"
            
            cv2.putText(
                frame, action_text,
                (w // 2 - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
            cv2.putText(
                frame, confidence_text,
                (w // 2 - 100 + len(action_text) * 12, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
            )
        elif action:
            action_text = f"Action: {action} (low confidence)"
            cv2.putText(
                frame, action_text,
                (w // 2 - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1
            )
        else:
            cv2.putText(
                frame, "Action: -- waiting --",
                (w // 2 - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1
            )
        
        return frame
    
    def run(
        self,
        camera_index: int = CAMERA_INDEX,
        save_video: bool = SAVE_VIDEO,
        save_dir: str = SAVE_VIDEO_DIR
    ):
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть камеру с индексом {camera_index}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nКамера открыта: {actual_width}x{actual_height}")
        print("Управление:")
        print("  'q' или 'ESC' - выход")
        print("  's' - сохранить текущий кадр")
        print("  'r' - сбросить буфер")
        print("  'm' - переключить отображение скелета")
        print("  'c' - переключить отображение соединений")
        print("-" * 50)
        
        video_writer = None
        recording = False
        
        fps_start_time = time.time()
        fps_counter = 0
        
        show_skeleton = SHOW_SKELETON
        show_connections = SHOW_CONNECTIONS
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Ошибка: не удалось получить кадр с камеры")
                    break
                
                frame = cv2.flip(frame, 1)
                
                start_time = time.time()
                
                action, confidence, landmarks = self.update(frame)
                
                if landmarks is not None and show_skeleton:
                    frame = self.draw_skeleton(
                        frame, landmarks, actual_width, actual_height
                    )
                
                frame = self.draw_info(frame, action, confidence, self.fps)
                
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    self.fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                if save_video:
                    if video_writer is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        os.makedirs(save_dir, exist_ok=True)
                        video_path = os.path.join(save_dir, f"recording_{timestamp}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(
                            video_path, fourcc, CAMERA_FPS,
                            (actual_width, actual_height)
                        )
                        print(f"Запись видео начата: {video_path}")
                    
                    if video_writer is not None:
                        video_writer.write(frame)
                
                cv2.imshow('Action Recognition - Real Time', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs("screenshots", exist_ok=True)
                    screenshot_path = os.path.join("screenshots", f"screenshot_{timestamp}.png")
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Скриншот сохранён: {screenshot_path}")
                
                elif key == ord('r'):
                    self.frame_buffer.clear()
                    self.prediction_buffer.clear()
                    print("Буфер сброшен")
                
                elif key == ord('m'):
                    show_skeleton = not show_skeleton
                    print(f"Отображение скелета: {'вкл' if show_skeleton else 'выкл'}")
                
                elif key == ord('c'):
                    show_connections = not show_connections
                    print(f"Отображение соединений: {'вкл' if show_connections else 'выкл'}")
                
                elapsed = time.time() - start_time
                if elapsed < 1.0 / CAMERA_FPS:
                    time.sleep(1.0 / CAMERA_FPS - elapsed)
        
        except KeyboardInterrupt:
            print("\nПрерывание пользователя")
        
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            self.pose_detector.close()
            print("Завершение работы")

def run_inference(
    model_type: str = MODEL_TYPE,
    camera_index: int = CAMERA_INDEX,
    save_video: bool = SAVE_VIDEO,
    checkpoint_dir: str = CHECKPOINT_DIR
):
    inference = RealTimeInference(
        model_type=model_type,
        checkpoint_dir=checkpoint_dir,
        window_size=WINDOW_SIZE,
        smoothing_window=SMOOTHING_WINDOW,
        confidence_threshold=CONFIDENCE_THRESHOLD_DISPLAY
    )
    
    inference.run(
        camera_index=camera_index,
        save_video=save_video,
        save_dir=SAVE_VIDEO_DIR
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс в реальном времени для распознавания действий")
    parser.add_argument(
        '--model', '-m', type=str, default=MODEL_TYPE,
        choices=['gru', 'cnn_lstm', 'tcn', 'stgcn'],
        help='Тип модели для использования'
    )
    parser.add_argument(
        '--camera', '-c', type=int, default=CAMERA_INDEX,
        help='Индекс веб-камеры'
    )
    parser.add_argument(
        '--save_video', '-s', action='store_true', default=SAVE_VIDEO,
        help='Сохранять видео с результатами'
    )
    parser.add_argument(
        '--checkpoint_dir', '-d', type=str, default=CHECKPOINT_DIR,
        help='Директория с чекпоинтами моделей'
    )
    parser.add_argument(
        '--window_size', '-w', type=int, default=WINDOW_SIZE,
        help='Размер временного окна (кадров)'
    )
    parser.add_argument(
        '--smoothing', type=int, default=SMOOTHING_WINDOW,
        help='Размер окна для сглаживания предсказаний'
    )
    parser.add_argument(
        '--confidence', type=float, default=CONFIDENCE_THRESHOLD_DISPLAY,
        help='Порог уверенности для отображения'
    )
    
    args = parser.parse_args()
    
    MODEL_TYPE = args.model
    CAMERA_INDEX = args.camera
    SAVE_VIDEO = args.save_video
    CHECKPOINT_DIR = args.checkpoint_dir
    WINDOW_SIZE = args.window_size
    SMOOTHING_WINDOW = args.smoothing
    CONFIDENCE_THRESHOLD_DISPLAY = args.confidence
    
    run_inference(
        model_type=MODEL_TYPE,
        camera_index=CAMERA_INDEX,
        save_video=SAVE_VIDEO,
        checkpoint_dir=CHECKPOINT_DIR
    )