import cv2
import numpy as np
import mediapipe as mp
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

MP_STATIC_IMAGE_MODE = False
MP_MODEL_COMPLEXITY = 1
MP_SMOOTH_LANDMARKS = True
MP_ENABLE_SEGMENTATION = False
MP_SMOOTH_SEGMENTATION = False
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

TARGET_FPS = 30
CONFIDENCE_THRESHOLD = 0.5

CAMERA_INDEX = 0

LANDMARK_INDICES = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'left_mouth': 9, 'right_mouth': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}

ROOT_JOINT_INDICES = [23, 24]
SHOULDER_INDICES = [11, 12]

def _init_pose_detector():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        static_image_mode=MP_STATIC_IMAGE_MODE,
        model_complexity=MP_MODEL_COMPLEXITY,
        smooth_landmarks=MP_SMOOTH_LANDMARKS,
        enable_segmentation=MP_ENABLE_SEGMENTATION,
        smooth_segmentation=MP_SMOOTH_SEGMENTATION,
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE
    )

def extract_landmarks_from_video(video_path: str, target_fps: int = TARGET_FPS,confidence_threshold: float = CONFIDENCE_THRESHOLD
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    print(f"Извлечение скелетных данных из видео: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видеофайл: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Исходное видео: {original_fps:.1f} FPS, {frame_count} кадров")
    
    pose_detector = _init_pose_detector()
    
    all_landmarks = []
    all_confidences = []
    
    frame_idx = 0
    processed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        if original_fps > target_fps:
            skip_ratio = original_fps / target_fps
            if frame_idx % int(skip_ratio) != 0:
                continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose_detector.process(frame_rgb)
        
        if result.pose_landmarks:
            frame_landmarks = []
            frame_confidences = []
            
            for i, lm in enumerate(result.pose_landmarks.landmark):
                frame_landmarks.append([lm.x, lm.y])
                confidence = getattr(lm, 'visibility', 1.0)
                frame_confidences.append(confidence)
            
            all_landmarks.append(frame_landmarks)
            all_confidences.append(frame_confidences)
        else:
            frame_landmarks = [[0.0, 0.0] for _ in range(33)]
            frame_confidences = [[0.0] for _ in range(33)]
            all_landmarks.append(frame_landmarks)
            all_confidences.append(frame_confidences)
        
        processed_frames += 1
        if processed_frames % 100 == 0:
            print(f"  Обработано кадров: {processed_frames}")
    
    cap.release()
    pose_detector.close()
    
    landmarks = np.array(all_landmarks, dtype=np.float32)
    confidences = np.array(all_confidences, dtype=np.float32)
    
    landmarks[confidences < confidence_threshold] = np.nan
    
    metadata = {
        'video_path': video_path,
        'original_fps': original_fps,
        'target_fps': target_fps,
        'original_frame_count': frame_count,
        'extracted_frame_count': processed_frames,
        'landmarks_shape': landmarks.shape,
        'confidence_threshold': confidence_threshold
    }
    
    print(f"  Извлечено {processed_frames} кадров. Формат скелета: {landmarks.shape}")
    
    return landmarks, confidences, metadata

def extract_landmarks_from_camera(camera_index: int = CAMERA_INDEX, callback=None, confidence_threshold: float = CONFIDENCE_THRESHOLD
):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть камеру с индексом {camera_index}")
    
    pose_detector = _init_pose_detector()
    
    print(f"Запуск веб-камеры (индекс {camera_index}). Нажмите 'q' для выхода.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose_detector.process(frame_rgb)
        
        landmarks = None
        confidences = None
        
        if result.pose_landmarks:
            frame_landmarks = []
            frame_confidences = []
            
            for i, lm in enumerate(result.pose_landmarks.landmark):
                frame_landmarks.append([lm.x, lm.y])
                confidence = getattr(lm, 'visibility', 1.0)
                frame_confidences.append(confidence)
            
            landmarks = np.array(frame_landmarks, dtype=np.float32)
            confidences = np.array(frame_confidences, dtype=np.float32)
            landmarks[confidences < confidence_threshold] = np.nan
        
        if callback is not None:
            callback(landmarks, confidences)
        
        mp.solutions.drawing_utils.draw_landmarks(
            frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
        )
        cv2.imshow('MediaPipe Pose - Skeleton Extraction', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    pose_detector.close()
    cv2.destroyAllWindows()
    print("Завершение работы с камерой")

def save_landmarks_to_file(
    landmarks: np.ndarray,
    confidences: np.ndarray,
    metadata: Dict[str, Any],
    output_path: str
):
    np.save(f"{output_path}_landmarks.npy", landmarks)
    np.save(f"{output_path}_confidences.npy", confidences)
    
    with open(f"{output_path}_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    print(f"Сохранено: {output_path}_landmarks.npy, {output_path}_confidences.npy, {output_path}_metadata.json")


def load_landmarks_from_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    landmarks = np.load(f"{filepath}_landmarks.npy")
    confidences = np.load(f"{filepath}_confidences.npy")
    
    with open(f"{filepath}_metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return landmarks, confidences, metadata
