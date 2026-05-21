import numpy as np
import random
from typing import Tuple, Optional, List, Dict, Any

PROB_HORIZONTAL_FLIP = 0.5
PROB_RANDOM_SCALE = 0.5 
PROB_RANDOM_ROTATION = 0.3

SCALE_MIN = 0.9
SCALE_MAX = 1.1

ROTATION_DEGREES_MIN = -10 
ROTATION_DEGREES_MAX = 10

ROOT_JOINT_INDICES = [23, 24]

SHOULDER_INDICES = [11, 12]

SYMMETRY_TYPE = 'swap_labels' # 'fixed'

SYMMETRY_MAP = {
    1: 4, 4: 1,
    2: 5, 5: 2,
    3: 6, 6: 3,
    7: 8, 8: 7,
    9: 10, 10: 9,
    11: 12, 12: 11,
    13: 14, 14: 13,
    15: 16, 16: 15,
    17: 18, 18: 17,
    19: 20, 20: 19,
    21: 22, 22: 21,
    23: 24, 24: 23,
    25: 26, 26: 25,
    27: 28, 28: 27,
    29: 30, 30: 29,
    31: 32, 32: 31,
}

EPSILON = 1e-6

def _get_root_center(landmarks: np.ndarray) -> Optional[np.ndarray]:
    root_points = []
    for idx in ROOT_JOINT_INDICES:
        if idx < len(landmarks):
            point = landmarks[idx]
            if not np.any(np.isnan(point)):
                root_points.append(point)
    
    if root_points:
        return np.mean(root_points, axis=0)
    else:
        return None

def horizontal_flip(
    landmarks: np.ndarray,
    symmetry_map: Dict[int, int] = None,
    symmetry_type: str = SYMMETRY_TYPE
) -> np.ndarray:
    if symmetry_map is None:
        symmetry_map = SYMMETRY_MAP
    
    is_single_frame = (landmarks.ndim == 2)
    if is_single_frame:
        landmarks = landmarks[np.newaxis, ...]
    
    flipped = landmarks.copy()
    n_frames = flipped.shape[0]
    
    for frame_idx in range(n_frames):
        frame = flipped[frame_idx]
        
        root_center = _get_root_center(frame)
        if root_center is None:
            valid_points = frame[~np.any(np.isnan(frame), axis=1)]
            if len(valid_points) > 0:
                root_center = np.mean(valid_points, axis=0)
            else:
                continue
        
        for i in range(len(frame)):
            if not np.any(np.isnan(frame[i])):
                frame[i, 0] = 2 * root_center[0] - frame[i, 0]
        
        if symmetry_type == 'swap_labels':
            new_frame = frame.copy()
            for left_idx, right_idx in symmetry_map.items():
                if left_idx < len(frame) and right_idx < len(frame):
                    temp_left = new_frame[left_idx].copy()
                    temp_right = new_frame[right_idx].copy()
                    new_frame[left_idx] = temp_right
                    new_frame[right_idx] = temp_left
            flipped[frame_idx] = new_frame
    
    if is_single_frame:
        flipped = flipped[0]
    
    return flipped


def random_scale(
    landmarks: np.ndarray,
    scale_min: float = SCALE_MIN,
    scale_max: float = SCALE_MAX,
    per_frame: bool = False,
    per_joint: bool = False
) -> np.ndarray:
    is_single_frame = (landmarks.ndim == 2)
    if is_single_frame:
        landmarks = landmarks[np.newaxis, ...]
    
    scaled = landmarks.copy()
    n_frames, n_joints, _ = scaled.shape
    
    for frame_idx in range(n_frames):
        frame = scaled[frame_idx]
        
        root_center = _get_root_center(frame)
        if root_center is None:
            continue
        
        if per_joint:
            scales = np.random.uniform(scale_min, scale_max, size=(n_joints, 1))
        elif per_frame:
            scales = np.random.uniform(scale_min, scale_max)
        else:
            scales = np.random.uniform(scale_min, scale_max)
        
        for i in range(n_joints):
            if not np.any(np.isnan(frame[i])):
                if isinstance(scales, np.ndarray):
                    scale_val = scales[i, 0] if per_joint else scales
                else:
                    scale_val = scales
                
                frame[i] = root_center + scale_val * (frame[i] - root_center)
    
    if is_single_frame:
        scaled = scaled[0]
    
    return scaled


def random_rotation(
    landmarks: np.ndarray,
    degrees_min: float = ROTATION_DEGREES_MIN,
    degrees_max: float = ROTATION_DEGREES_MAX,
    per_frame: bool = False
) -> np.ndarray:
    is_single_frame = (landmarks.ndim == 2)
    if is_single_frame:
        landmarks = landmarks[np.newaxis, ...]
    
    rotated = landmarks.copy()
    n_frames = rotated.shape[0]
    
    for frame_idx in range(n_frames):
        frame = rotated[frame_idx]
        
        root_center = _get_root_center(frame)
        if root_center is None:
            continue
        
        if per_frame:
            angle = np.random.uniform(degrees_min, degrees_max)
        else:
            angle = np.random.uniform(degrees_min, degrees_max)
        
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        for i in range(len(frame)):
            if not np.any(np.isnan(frame[i])):
                translated = frame[i] - root_center
                rotated_point = np.dot(rotation_matrix, translated)
                frame[i] = rotated_point + root_center
    
    if is_single_frame:
        rotated = rotated[0]
    
    return rotated


def apply_spatial_augmentation(
    landmarks: np.ndarray,
    prob_flip: float = PROB_HORIZONTAL_FLIP,
    prob_scale: float = PROB_RANDOM_SCALE,
    prob_rotation: float = PROB_RANDOM_ROTATION,
    scale_min: float = SCALE_MIN,
    scale_max: float = SCALE_MAX,
    rotation_min: float = ROTATION_DEGREES_MIN,
    rotation_max: float = ROTATION_DEGREES_MAX
) -> np.ndarray:
    result = landmarks.copy()
    
    if random.random() < prob_flip:
        result = horizontal_flip(result)
    
    if random.random() < prob_scale:
        result = random_scale(result, scale_min, scale_max)
    
    if random.random() < prob_rotation:
        result = random_rotation(result, rotation_min, rotation_max)
    
    return result

class SpatialAugmenter:
    def __init__(
        self,
        prob_flip: float = PROB_HORIZONTAL_FLIP,
        prob_scale: float = PROB_RANDOM_SCALE,
        prob_rotation: float = PROB_RANDOM_ROTATION,
        scale_min: float = SCALE_MIN,
        scale_max: float = SCALE_MAX,
        rotation_min: float = ROTATION_DEGREES_MIN,
        rotation_max: float = ROTATION_DEGREES_MAX
    ):
        self.prob_flip = prob_flip
        self.prob_scale = prob_scale
        self.prob_rotation = prob_rotation
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rotation_min = rotation_min
        self.rotation_max = rotation_max
    
    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        return apply_spatial_augmentation(
            landmarks,
            self.prob_flip,
            self.prob_scale,
            self.prob_rotation,
            self.scale_min,
            self.scale_max,
            self.rotation_min,
            self.rotation_max
        )
    
    def augment_batch(self, batch: np.ndarray) -> np.ndarray:
        batch_size = batch.shape[0]
        augmented = []
        
        for i in range(batch_size):
            aug_sample = self(batch[i])
            augmented.append(aug_sample)
        
        return np.array(augmented)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'prob_flip': self.prob_flip,
            'prob_scale': self.prob_scale,
            'prob_rotation': self.prob_rotation,
            'scale_min': self.scale_min,
            'scale_max': self.scale_max,
            'rotation_min': self.rotation_min,
            'rotation_max': self.rotation_max
        }
