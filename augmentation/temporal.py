import numpy as np
import random
from typing import Tuple, Optional, List, Dict, Any
from scipy import interpolate

PROB_TEMPORAL_SHIFT = 0.4
PROB_TEMPORAL_STRETCH = 0.3

SHIFT_MIN_FRAMES = -10
SHIFT_MAX_FRAMES = 10

STRETCH_MIN = 0.8
STRETCH_MAX = 1.2

SHIFT_BOUNDARY_STRATEGY = 'pad' # 'wrap' или 'drop'

PAD_TYPE = 'zero' # 'first' или 'last'

STRETCH_INTERPOLATION = 'linear' # 'nearest'

def temporal_shift(
    landmarks: np.ndarray,
    shift_frames: int = None,
    shift_min: int = SHIFT_MIN_FRAMES,
    shift_max: int = SHIFT_MAX_FRAMES,
    boundary_strategy: str = SHIFT_BOUNDARY_STRATEGY,
    pad_type: str = PAD_TYPE
) -> np.ndarray:
    n_frames = landmarks.shape[0]
    
    if shift_frames is None:
        shift_frames = random.randint(shift_min, shift_max)
    
    if shift_frames == 0:
        return landmarks.copy()
    
    shifted = np.zeros_like(landmarks)
    
    if boundary_strategy == 'wrap':
        for t in range(n_frames):
            source_t = (t - shift_frames) % n_frames
            shifted[t] = landmarks[source_t]
    
    elif boundary_strategy == 'pad':
        for t in range(n_frames):
            source_t = t - shift_frames
            if 0 <= source_t < n_frames:
                shifted[t] = landmarks[source_t]
            else:
                if pad_type == 'zero':
                    shifted[t] = np.zeros_like(landmarks[0])
                elif pad_type == 'first':
                    shifted[t] = landmarks[0]
                elif pad_type == 'last':
                    shifted[t] = landmarks[-1]
                else:
                    shifted[t] = np.zeros_like(landmarks[0])
    
    elif boundary_strategy == 'drop':
        if shift_frames > 0:
            shifted[:-shift_frames] = landmarks[shift_frames:]
        elif shift_frames < 0:
            shifted[-shift_frames:] = landmarks[:shift_frames]
    
    return shifted


def temporal_stretch(
    landmarks: np.ndarray,
    stretch_factor: float = None,
    stretch_min: float = STRETCH_MIN,
    stretch_max: float = STRETCH_MAX,
    interpolation: str = STRETCH_INTERPOLATION
) -> np.ndarray:
    n_frames = landmarks.shape[0]
    
    if stretch_factor is None:
        stretch_factor = random.uniform(stretch_min, stretch_max)
    
    if abs(stretch_factor - 1.0) < 0.01:
        return landmarks.copy()
    
    new_length = int(n_frames * stretch_factor)
    
    if new_length <= 0:
        new_length = 1
    
    old_indices = np.arange(n_frames)
    new_indices = np.linspace(0, n_frames - 1, new_length)
    
    stretched = np.zeros((new_length, landmarks.shape[1], landmarks.shape[2]), dtype=landmarks.dtype)
    
    for joint_idx in range(landmarks.shape[1]):
        for coord_idx in range(landmarks.shape[2]):
            old_values = landmarks[:, joint_idx, coord_idx]
            
            valid_mask = ~np.isnan(old_values)
            if np.any(valid_mask):
                if interpolation == 'linear':
                    interp_func = interpolate.interp1d(
                        old_indices[valid_mask],
                        old_values[valid_mask],
                        kind='linear',
                        fill_value='extrapolate'
                    )
                    new_values = interp_func(new_indices)
                else:
                    interp_func = interpolate.interp1d(
                        old_indices[valid_mask],
                        old_values[valid_mask],
                        kind='nearest',
                        fill_value='extrapolate'
                    )
                    new_values = interp_func(new_indices)
                
                stretched[:, joint_idx, coord_idx] = new_values
            else:
                stretched[:, joint_idx, coord_idx] = 0.0
    
    return stretched


def apply_temporal_augmentation(
    landmarks: np.ndarray,
    prob_shift: float = PROB_TEMPORAL_SHIFT,
    prob_stretch: float = PROB_TEMPORAL_STRETCH,
    shift_min: int = SHIFT_MIN_FRAMES,
    shift_max: int = SHIFT_MAX_FRAMES,
    stretch_min: float = STRETCH_MIN,
    stretch_max: float = STRETCH_MAX
) -> np.ndarray:
    result = landmarks.copy()
    
    if random.random() < prob_shift:
        result = temporal_shift(result, shift_min=shift_min, shift_max=shift_max)
    
    if random.random() < prob_stretch:
        result = temporal_stretch(result, stretch_min=stretch_min, stretch_max=stretch_max)
    
    return result

class TemporalAugmenter:
    def __init__(
        self,
        prob_shift: float = PROB_TEMPORAL_SHIFT,
        prob_stretch: float = PROB_TEMPORAL_STRETCH,
        shift_min: int = SHIFT_MIN_FRAMES,
        shift_max: int = SHIFT_MAX_FRAMES,
        stretch_min: float = STRETCH_MIN,
        stretch_max: float = STRETCH_MAX
    ):
        self.prob_shift = prob_shift
        self.prob_stretch = prob_stretch
        self.shift_min = shift_min
        self.shift_max = shift_max
        self.stretch_min = stretch_min
        self.stretch_max = stretch_max
    
    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        return apply_temporal_augmentation(
            landmarks,
            self.prob_shift,
            self.prob_stretch,
            self.shift_min,
            self.shift_max,
            self.stretch_min,
            self.stretch_max
        )
    
    def augment_batch(self, batch: np.ndarray) -> np.ndarray:
        batch_size = batch.shape[0]
        augmented = []
        
        for i in range(batch_size):
            aug_sample = self(batch[i])
            augmented.append(aug_sample)
        
        max_len = max(seq.shape[0] for seq in augmented)
        padded_batch = []
        for seq in augmented:
            if seq.shape[0] < max_len:
                pad = np.zeros((max_len - seq.shape[0], seq.shape[1], seq.shape[2]))
                seq = np.concatenate([seq, pad], axis=0)
            padded_batch.append(seq)
        
        return np.array(padded_batch)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'prob_shift': self.prob_shift,
            'prob_stretch': self.prob_stretch,
            'shift_min': self.shift_min,
            'shift_max': self.shift_max,
            'stretch_min': self.stretch_min,
            'stretch_max': self.stretch_max
        }
