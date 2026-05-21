import numpy as np
from typing import Tuple, Optional

ROOT_JOINT_INDICES = [23, 24]

SHOULDER_INDICES = [11, 12]

SCALE_TYPE = 'shoulder_distance' # 'height' или 'none'

EPSILON = 1e-6

def normalize_skeleton(
    landmarks: np.ndarray,
    root_indices: list = None,
    shoulder_indices: list = None,
    scale_type: str = None,
    epsilon: float = EPSILON
) -> np.ndarray:
    if root_indices is None:
        root_indices = ROOT_JOINT_INDICES
    if shoulder_indices is None:
        shoulder_indices = SHOULDER_INDICES
    if scale_type is None:
        scale_type = SCALE_TYPE
    
    normalized = landmarks.copy()
    
    n_frames = normalized.shape[0]
    
    for frame_idx in range(n_frames):
        frame_landmarks = normalized[frame_idx]
        root_points = []
        for idx in root_indices:
            if idx < len(frame_landmarks):
                point = frame_landmarks[idx]
                if not np.any(np.isnan(point)):
                    root_points.append(point)
        
        if len(root_points) > 0:
            root_center = np.mean(root_points, axis=0)
        else:
            continue

        for i in range(len(frame_landmarks)):
            if not np.any(np.isnan(frame_landmarks[i])):
                frame_landmarks[i] = frame_landmarks[i] - root_center
        
        if scale_type == 'shoulder_distance':
            shoulder_left = frame_landmarks[shoulder_indices[0]]
            shoulder_right = frame_landmarks[shoulder_indices[1]]
            
            if not np.any(np.isnan(shoulder_left)) and not np.any(np.isnan(shoulder_right)):
                D = np.linalg.norm(shoulder_right - shoulder_left)
            else:
                valid_points = frame_landmarks[~np.any(np.isnan(frame_landmarks), axis=1)]
                if len(valid_points) > 1:
                    D = np.max(valid_points[:, 0]) - np.min(valid_points[:, 0])
                else:
                    D = 1.0
            
            D = max(D, epsilon)
            
            for i in range(len(frame_landmarks)):
                if not np.any(np.isnan(frame_landmarks[i])):
                    frame_landmarks[i] = frame_landmarks[i] / D
        
        elif scale_type == 'height':
            head_idx = 0
            foot_indices = [27, 28]
            
            head = frame_landmarks[head_idx]
            foot_points = []
            for idx in foot_indices:
                if idx < len(frame_landmarks) and not np.any(np.isnan(frame_landmarks[idx])):
                    foot_points.append(frame_landmarks[idx])
            
            if len(foot_points) > 0 and not np.any(np.isnan(head)):
                foot_center = np.mean(foot_points, axis=0)
                height = abs(foot_center[1] - head[1])
            else:
                height = 1.0
            
            height = max(height, epsilon)
            
            for i in range(len(frame_landmarks)):
                if not np.any(np.isnan(frame_landmarks[i])):
                    frame_landmarks[i] = frame_landmarks[i] / height
        
    return normalized


def normalize_batch(
    landmarks_batch: np.ndarray,
    root_indices: list = None,
    shoulder_indices: list = None,
    scale_type: str = None
) -> np.ndarray:
    batch_size = landmarks_batch.shape[0]
    normalized_batch = []
    
    for i in range(batch_size):
        normalized_seq = normalize_skeleton(
            landmarks_batch[i], 
            root_indices, 
            shoulder_indices, 
            scale_type
        )
        normalized_batch.append(normalized_seq)
    
    return np.array(normalized_batch)


def calculate_scale_factor(landmarks: np.ndarray, shoulder_indices: list = None) -> float:
    if shoulder_indices is None:
        shoulder_indices = SHOULDER_INDICES
    
    shoulder_left = landmarks[:, shoulder_indices[0], :]
    shoulder_right = landmarks[:, shoulder_indices[1], :]
    
    distances = []
    for frame_idx in range(landmarks.shape[0]):
        left = shoulder_left[frame_idx]
        right = shoulder_right[frame_idx]
        if not np.any(np.isnan(left)) and not np.any(np.isnan(right)):
            dist = np.linalg.norm(right - left)
            distances.append(dist)
    
    if distances:
        return np.mean(distances)
    else:
        return 1.0
