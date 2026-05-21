import numpy as np
from typing import Optional

SMOOTHING_ALPHA = 0.3

ALPHA_MIN = 0.1

ALPHA_MAX = 0.7

def exponential_smoothing(
    landmarks: np.ndarray,
    alpha: float = SMOOTHING_ALPHA,
    axis: int = 0
) -> np.ndarray:
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha должен быть в диапазоне (0, 1], получено {alpha}")
    
    smoothed = landmarks.copy()
    
    for t in range(1, smoothed.shape[axis]):
        smoothed[t] = alpha * landmarks[t] + (1 - alpha) * smoothed[t - 1]
    
    return smoothed


def smooth_batch(
    landmarks_batch: np.ndarray,
    alpha: float = SMOOTHING_ALPHA
) -> np.ndarray:
    batch_size = landmarks_batch.shape[0]
    smoothed_batch = []
    
    for i in range(batch_size):
        smoothed_seq = exponential_smoothing(landmarks_batch[i], alpha)
        smoothed_batch.append(smoothed_seq)
    
    return np.array(smoothed_batch)


def adaptive_smoothing(landmarks: np.ndarray, alpha_min: float = ALPHA_MIN, alpha_max: float = ALPHA_MAX) -> np.ndarray:
    smoothed = landmarks.copy()
    
    velocity = np.abs(np.diff(landmarks, axis=0))
    velocity_mean = np.mean(velocity, axis=(1, 2))
    
    if len(velocity_mean) > 0:
        v_min = np.min(velocity_mean)
        v_max = np.max(velocity_mean)
        
        if v_max > v_min:
            velocity_norm = (velocity_mean - v_min) / (v_max - v_min)
        else:
            velocity_norm = np.zeros_like(velocity_mean)
    else:
        velocity_norm = np.zeros(landmarks.shape[0] - 1)
    
    for t in range(1, smoothed.shape[0]):
        alpha = alpha_min + velocity_norm[t - 1] * (alpha_max - alpha_min)
        smoothed[t] = alpha * landmarks[t] + (1 - alpha) * smoothed[t - 1]
    
    return smoothed


def get_optimal_alpha(landmarks: np.ndarray) -> float:
    diffs = np.diff(landmarks, axis=0)
    
    noise_estimate = np.median(np.abs(diffs), axis=0)
    noise_level = np.mean(noise_estimate)
    
    signal_range = np.max(landmarks, axis=0) - np.min(landmarks, axis=0)
    signal_level = np.mean(signal_range)
    
    if signal_level > 0:
        snr = signal_level / (noise_level + 1e-6)
    else:
        snr = 1.0
    

    alpha = max(0.1, min(0.5, 1.0 / (1.0 + snr / 10.0)))
    
    return alpha
