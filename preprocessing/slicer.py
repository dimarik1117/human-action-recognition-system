import numpy as np
from typing import Tuple, List, Generator, Optional

WINDOW_SIZE = 30

STRIDE = 15

END_STRATEGY = 'pad' # 'drop'

PAD_TYPE = 'zero' # 'last'

def sliding_window_slice(
    landmarks: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    end_strategy: str = END_STRATEGY,
    pad_type: str = PAD_TYPE
) -> np.ndarray:
    n_frames = landmarks.shape[0]
    
    flat_landmarks = landmarks.reshape(n_frames, -1)
    
    windows = []
    
    start = 0
    while start + window_size <= n_frames:
        window = flat_landmarks[start:start + window_size]
        windows.append(window)
        start += stride
    
    if start < n_frames and end_strategy == 'pad':
        remaining = n_frames - start
        window = flat_landmarks[start:n_frames]
        
        if pad_type == 'zero':
            pad_width = window_size - remaining
            pad = np.zeros((pad_width, flat_landmarks.shape[1]))
        elif pad_type == 'last':
            pad_width = window_size - remaining
            last_value = flat_landmarks[-1] if n_frames > 0 else 0
            pad = np.tile(last_value, (pad_width, 1))
        else:
            raise ValueError(f"Неизвестный тип паддинга: {pad_type}")
        
        window = np.vstack([window, pad])
        windows.append(window)
    
    windows = np.array(windows, dtype=np.float32)
    
    return windows


def sliding_window_slice_with_labels(
    landmarks: np.ndarray,
    labels: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    end_strategy: str = END_STRATEGY,
    label_policy: str = 'majority'
) -> Tuple[np.ndarray, np.ndarray]:
    n_frames = landmarks.shape[0]
    flat_landmarks = landmarks.reshape(n_frames, -1)
    
    windows = []
    window_labels = []
    
    start = 0
    while start + window_size <= n_frames:
        window = flat_landmarks[start:start + window_size]
        window_label = _get_window_label(labels[start:start + window_size], label_policy)
        windows.append(window)
        window_labels.append(window_label)
        start += stride
    
    if start < n_frames and end_strategy == 'pad':
        remaining = n_frames - start
        window = flat_landmarks[start:n_frames]
        
        pad_width = window_size - remaining
        pad = np.zeros((pad_width, flat_landmarks.shape[1]))
        window = np.vstack([window, pad])
        
        label_window = labels[start:n_frames]
        label_pad = np.full(pad_width, -1)
        label_window = np.concatenate([label_window, label_pad])
        window_label = _get_window_label(label_window, label_policy)
        
        windows.append(window)
        window_labels.append(window_label)
    
    windows = np.array(windows, dtype=np.float32)
    windows = windows.reshape(-1, window_size, 33, 2)
    window_labels = np.array(window_labels, dtype=np.int64)
    
    return windows, window_labels


def _get_window_label(labels_window: np.ndarray, policy: str) -> int:
    valid_labels = labels_window[labels_window >= 0]
    
    if len(valid_labels) == 0:
        return -1
    
    if policy == 'first':
        return int(valid_labels[0])
    elif policy == 'last':
        return int(valid_labels[-1])
    elif policy == 'center':
        center = len(valid_labels) // 2
        return int(valid_labels[center])
    elif policy == 'majority':
        from collections import Counter
        counter = Counter(valid_labels)
        return int(counter.most_common(1)[0][0])
    else:
        raise ValueError(f"Неизвестный policy: {policy}")


class SlidingWindowGenerator:
    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        buffer_size: Optional[int] = None
    ):
        self.window_size = window_size
        self.stride = stride
        self.buffer_size = buffer_size if buffer_size is not None else window_size
        
        self.buffer = []
        self.frame_counter = 0
    
    def add_frame(self, frame_landmarks: np.ndarray) -> List[np.ndarray]:
        self.buffer.append(frame_landmarks.flatten())
        self.frame_counter += 1
        
        windows = []
        
        if len(self.buffer) >= self.window_size:
            if (self.frame_counter - self.window_size) % self.stride == 0:
                window = np.array(self.buffer[-self.window_size:], dtype=np.float32)
                windows.append(window)
        
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        
        return windows
    
    def get_ready_windows(self) -> List[np.ndarray]:
        windows = []
        
        if len(self.buffer) >= self.window_size:
            start_idx = 0
            while start_idx + self.window_size <= len(self.buffer):
                window = np.array(self.buffer[start_idx:start_idx + self.window_size], dtype=np.float32)
                windows.append(window)
                start_idx += self.stride
        
        return windows
    
    def reset(self):
        self.buffer = []
        self.frame_counter = 0


def get_num_windows(n_frames: int, window_size: int = WINDOW_SIZE, stride: int = STRIDE) -> int:
    if n_frames < window_size:
        return 1
    
    num_windows = (n_frames - window_size) // stride + 1
    return max(1, num_windows)
