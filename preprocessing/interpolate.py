import numpy as np
from typing import Tuple, Optional

MAX_GAP_FOR_INTERPOLATION = 10

LONG_GAP_STRATEGY = 'symmetry' # 'zeros' или 'constant'

CONSTANT_FILL_VALUE = 0.0

SYMMETRIC_JOINT_PAIRS = [
    (11, 12),
    (13, 14),
    (15, 16),
    (23, 24),
    (25, 26),
    (27, 28),
    (7, 8),
    (1, 4),
    (2, 5),
    (3, 6),
    (9, 10),
]

CENTER_AXIS_INDICES = [23, 24, 11, 12]

def interpolate_missing_points(
    landmarks: np.ndarray,
    max_gap: int = MAX_GAP_FOR_INTERPOLATION,
    long_gap_strategy: str = LONG_GAP_STRATEGY,
    symmetric_pairs: list = None,
    center_indices: list = None
) -> np.ndarray:
    if symmetric_pairs is None:
        symmetric_pairs = SYMMETRIC_JOINT_PAIRS
    if center_indices is None:
        center_indices = CENTER_AXIS_INDICES
    
    interpolated = landmarks.copy()
    n_frames, n_joints, n_coords = interpolated.shape
    
    for joint_idx in range(n_joints):
        for coord_idx in range(n_coords):
            series = interpolated[:, joint_idx, coord_idx]
            
            missing_mask = np.isnan(series)
            
            if not np.any(missing_mask):
                continue
            
            gaps = _find_gaps(missing_mask)
            
            for start, end in gaps:
                gap_length = end - start + 1
                
                left_val = series[start - 1] if start > 0 else None
                right_val = series[end + 1] if end < len(series) - 1 else None
                
                if gap_length <= max_gap and left_val is not None and right_val is not None:
                    for t in range(start, end + 1):
                        t_norm = (t - (start - 1)) / (gap_length + 1)
                        series[t] = left_val + t_norm * (right_val - left_val)
                
                else:
                    if long_gap_strategy == 'symmetry':
                        filled = _fill_by_symmetry(
                            interpolated, joint_idx, coord_idx, start, end,
                            symmetric_pairs, center_indices
                        )
                        if filled is not None:
                            for t in range(start, end + 1):
                                series[t] = filled
                            continue
                    
                    if long_gap_strategy == 'zeros':
                        series[start:end + 1] = 0.0
                    elif long_gap_strategy == 'constant':
                        if left_val is not None:
                            series[start:end + 1] = left_val
                        elif right_val is not None:
                            series[start:end + 1] = right_val
                        else:
                            series[start:end + 1] = CONSTANT_FILL_VALUE
    
    return interpolated


def _find_gaps(missing_mask: np.ndarray) -> list:
    gaps = []
    in_gap = False
    start = 0
    
    for i, is_missing in enumerate(missing_mask):
        if is_missing and not in_gap:
            start = i
            in_gap = True
        elif not is_missing and in_gap:
            gaps.append((start, i - 1))
            in_gap = False
    
    if in_gap:
        gaps.append((start, len(missing_mask) - 1))
    
    return gaps


def _fill_by_symmetry(
    landmarks: np.ndarray,
    joint_idx: int,
    coord_idx: int,
    gap_start: int,
    gap_end: int,
    symmetric_pairs: list,
    center_indices: list
) -> Optional[float]:
    symmetric_joint = None
    for left, right in symmetric_pairs:
        if joint_idx == left:
            symmetric_joint = right
            break
        elif joint_idx == right:
            symmetric_joint = left
            break
    
    if symmetric_joint is None:
        return None
    
    mid_frame = (gap_start + gap_end) // 2
    
    sym_series_x = landmarks[:, symmetric_joint, 0]
    sym_series_y = landmarks[:, symmetric_joint, 1]
    
    if np.isnan(sym_series_x[mid_frame]) or np.isnan(sym_series_y[mid_frame]):
        return None
    
    center_x = _compute_center_axis(landmarks, mid_frame, center_indices)
    if center_x is None:
        return None
    
    sym_x = sym_series_x[mid_frame]
    reflected_x = 2 * center_x - sym_x
    
    if coord_idx == 0:
        return reflected_x
    else:
        return sym_series_y[mid_frame]


def _compute_center_axis(
    landmarks: np.ndarray,
    frame_idx: int,
    center_indices: list
) -> Optional[float]:
    valid_x = []
    for idx in center_indices:
        x_val = landmarks[frame_idx, idx, 0]
        if not np.isnan(x_val):
            valid_x.append(x_val)
    
    if valid_x:
        return np.mean(valid_x)
    else:
        return None


def interpolate_batch(
    landmarks_batch: np.ndarray,
    max_gap: int = MAX_GAP_FOR_INTERPOLATION,
    long_gap_strategy: str = LONG_GAP_STRATEGY
) -> np.ndarray:
    batch_size = landmarks_batch.shape[0]
    interpolated_batch = []
    
    for i in range(batch_size):
        interpolated_seq = interpolate_missing_points(
            landmarks_batch[i], max_gap, long_gap_strategy
        )
        interpolated_batch.append(interpolated_seq)
    
    return np.array(interpolated_batch)
