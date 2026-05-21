import numpy as np
import random
from typing import Tuple, Optional, List, Dict, Any, Union

PROB_GAUSSIAN_NOISE = 0.6
PROB_JOINT_DROPOUT = 0.3
PROB_JITTER = 0.4

NOISE_MEAN = 0.0
NOISE_STD_MIN = 0.005
NOISE_STD_MAX = 0.02

DROPOUT_PROB_MIN = 0.05
DROPOUT_PROB_MAX = 0.15

JITTER_STD_MIN = 0.001
JITTER_STD_MAX = 0.005

NOISE_TYPE = 'additive' # 'multiplicative'

def add_gaussian_noise(
    landmarks: np.ndarray,
    mean: float = NOISE_MEAN,
    std_min: float = NOISE_STD_MIN,
    std_max: float = NOISE_STD_MAX,
    per_frame: bool = False,
    per_joint: bool = False,
    noise_type: str = NOISE_TYPE
) -> np.ndarray:
    is_single_frame = (landmarks.ndim == 2)
    if is_single_frame:
        landmarks = landmarks[np.newaxis, ...]
    
    noisy = landmarks.copy()
    n_frames, n_joints, n_coords = noisy.shape
    
    if per_joint:
        std = np.random.uniform(std_min, std_max, size=(n_frames, n_joints, 1))
    elif per_frame:
        std = np.random.uniform(std_min, std_max, size=(n_frames, 1, 1))
    else:
        std = np.random.uniform(std_min, std_max)
    
    for frame_idx in range(n_frames):
        for joint_idx in range(n_joints):
            for coord_idx in range(n_coords):
                if not np.isnan(noisy[frame_idx, joint_idx, coord_idx]):
                    if isinstance(std, np.ndarray):
                        if per_joint:
                            current_std = std[frame_idx, joint_idx, 0]
                        elif per_frame:
                            current_std = std[frame_idx, 0, 0]
                        else:
                            current_std = std
                    else:
                        current_std = std
                    
                    noise = np.random.normal(mean, current_std)
                    
                    if noise_type == 'additive':
                        noisy[frame_idx, joint_idx, coord_idx] += noise
                    else:
                        noisy[frame_idx, joint_idx, coord_idx] *= (1 + noise)
    
    if is_single_frame:
        noisy = noisy[0]
    
    return noisy


def random_joint_dropout(
    landmarks: np.ndarray,
    dropout_prob_min: float = DROPOUT_PROB_MIN,
    dropout_prob_max: float = DROPOUT_PROB_MAX,
    per_frame: bool = True
) -> np.ndarray:
    is_single_frame = (landmarks.ndim == 2)
    if is_single_frame:
        landmarks = landmarks[np.newaxis, ...]
    
    dropped = landmarks.copy()
    n_frames, n_joints, n_coords = dropped.shape
    
    for frame_idx in range(n_frames):
        if per_frame:
            dropout_prob = np.random.uniform(dropout_prob_min, dropout_prob_max)
        else:
            dropout_prob = np.random.uniform(dropout_prob_min, dropout_prob_max)
        
        for joint_idx in range(n_joints):
            if random.random() < dropout_prob:
                dropped[frame_idx, joint_idx] = [np.nan, np.nan]
    
    if is_single_frame:
        dropped = dropped[0]
    
    return dropped


def add_jitter(
    landmarks: np.ndarray,
    jitter_std_min: float = JITTER_STD_MIN,
    jitter_std_max: float = JITTER_STD_MAX,
    per_frame: bool = True
) -> np.ndarray:
    is_single_frame = (landmarks.ndim == 2)
    if is_single_frame:
        landmarks = landmarks[np.newaxis, ...]
    
    jittered = landmarks.copy()
    n_frames, n_joints, n_coords = jittered.shape
    
    for frame_idx in range(n_frames):
        if per_frame:
            jitter_std = np.random.uniform(jitter_std_min, jitter_std_max)
        else:
            jitter_std = np.random.uniform(jitter_std_min, jitter_std_max)
        
        for joint_idx in range(n_joints):
            for coord_idx in range(n_coords):
                if not np.isnan(jittered[frame_idx, joint_idx, coord_idx]):
                    jitter = np.random.normal(0, jitter_std)
                    jittered[frame_idx, joint_idx, coord_idx] += jitter
    
    if is_single_frame:
        jittered = jittered[0]
    
    return jittered


def apply_noise_augmentation(
    landmarks: np.ndarray,
    prob_noise: float = PROB_GAUSSIAN_NOISE,
    prob_dropout: float = PROB_JOINT_DROPOUT,
    prob_jitter: float = PROB_JITTER,
    noise_std_min: float = NOISE_STD_MIN,
    noise_std_max: float = NOISE_STD_MAX,
    dropout_prob_min: float = DROPOUT_PROB_MIN,
    dropout_prob_max: float = DROPOUT_PROB_MAX,
    jitter_std_min: float = JITTER_STD_MIN,
    jitter_std_max: float = JITTER_STD_MAX
) -> np.ndarray:
    result = landmarks.copy()
    
    if random.random() < prob_noise:
        result = add_gaussian_noise(result, std_min=noise_std_min, std_max=noise_std_max)
    
    if random.random() < prob_dropout:
        result = random_joint_dropout(result, dropout_prob_min=dropout_prob_min, dropout_prob_max=dropout_prob_max)
    
    if random.random() < prob_jitter:
        result = add_jitter(result, jitter_std_min=jitter_std_min, jitter_std_max=jitter_std_max)
    
    return result

class NoiseAugmenter:
    def __init__(
        self,
        prob_noise: float = PROB_GAUSSIAN_NOISE,
        prob_dropout: float = PROB_JOINT_DROPOUT,
        prob_jitter: float = PROB_JITTER,
        noise_std_min: float = NOISE_STD_MIN,
        noise_std_max: float = NOISE_STD_MAX,
        dropout_prob_min: float = DROPOUT_PROB_MIN,
        dropout_prob_max: float = DROPOUT_PROB_MAX,
        jitter_std_min: float = JITTER_STD_MIN,
        jitter_std_max: float = JITTER_STD_MAX
    ):
        self.prob_noise = prob_noise
        self.prob_dropout = prob_dropout
        self.prob_jitter = prob_jitter
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.dropout_prob_min = dropout_prob_min
        self.dropout_prob_max = dropout_prob_max
        self.jitter_std_min = jitter_std_min
        self.jitter_std_max = jitter_std_max
    
    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        return apply_noise_augmentation(
            landmarks,
            self.prob_noise,
            self.prob_dropout,
            self.prob_jitter,
            self.noise_std_min,
            self.noise_std_max,
            self.dropout_prob_min,
            self.dropout_prob_max,
            self.jitter_std_min,
            self.jitter_std_max
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
            'prob_noise': self.prob_noise,
            'prob_dropout': self.prob_dropout,
            'prob_jitter': self.prob_jitter,
            'noise_std_min': self.noise_std_min,
            'noise_std_max': self.noise_std_max,
            'dropout_prob_min': self.dropout_prob_min,
            'dropout_prob_max': self.dropout_prob_max,
            'jitter_std_min': self.jitter_std_min,
            'jitter_std_max': self.jitter_std_max
        }

def apply_augmentation_to_batch(
    batch: np.ndarray,
    augmenter
) -> np.ndarray:
    batch_size = batch.shape[0]
    augmented = []
    
    for i in range(batch_size):
        aug_sample = augmenter(batch[i])
        augmented.append(aug_sample)
    
    return np.array(augmented)
