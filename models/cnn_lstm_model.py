import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

NUM_JOINTS = 33
NUM_COORDS = 2
CNN_HIDDEN_DIMS = [64, 128]
CNN_KERNEL_SIZES = [3, 3]
CNN_STRIDES = [1, 1]
CNN_PADDING = [1, 1]

LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_BIDIRECTIONAL = False

FC_HIDDEN_SIZE = 64
NUM_CLASSES = 10

DROPOUT = 0.3

class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        num_joints: int = NUM_JOINTS,
        num_coords: int = NUM_COORDS,
        cnn_hidden_dims: List[int] = None,
        cnn_kernel_sizes: List[int] = None,
        lstm_hidden_size: int = LSTM_HIDDEN_SIZE,
        lstm_num_layers: int = LSTM_NUM_LAYERS,
        lstm_dropout: float = LSTM_DROPOUT,
        lstm_bidirectional: bool = LSTM_BIDIRECTIONAL,
        fc_hidden_size: int = FC_HIDDEN_SIZE,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT
    ):
        super(CNNLSTMModel, self).__init__()
        
        if cnn_hidden_dims is None:
            cnn_hidden_dims = CNN_HIDDEN_DIMS
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = CNN_KERNEL_SIZES
        
        self.num_joints = num_joints
        self.num_coords = num_coords
        self.cnn_hidden_dims = cnn_hidden_dims
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.num_directions = 2 if lstm_bidirectional else 1
        self.num_classes = num_classes
        
        cnn_layers = []
        in_channels = num_joints * num_coords
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_hidden_dims, cnn_kernel_sizes)):
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=CNN_STRIDES[i] if i < len(CNN_STRIDES) else 1,
                    padding=CNN_PADDING[i] if i < len(CNN_PADDING) else 0
                )
            )
            cnn_layers.append(nn.BatchNorm1d(out_channels))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        cnn_output_size = cnn_hidden_dims[-1] if cnn_hidden_dims else 66
        
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=lstm_bidirectional
        )
        
        lstm_output_size = lstm_hidden_size * self.num_directions
        
        if fc_hidden_size > 0:
            self.fc = nn.Sequential(
                nn.Linear(lstm_output_size, fc_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden_size, num_classes)
            )
        else:
            self.fc = nn.Linear(lstm_output_size, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, num_joints, num_coords = x.shape
        
        x = x.reshape(batch_size, seq_len, num_joints * num_coords)
        
        x = x.transpose(1, 2)
        
        x = self.cnn(x)
        
        x = x.transpose(1, 2)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.lstm_bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            features = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            features = hidden[-1, :, :]
        
        if return_features:
            return features
        
        out = self.fc(features)
        
        return out
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return probs, preds
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'model_type': 'CNN+LSTM',
            'num_joints': self.num_joints,
            'num_coords': self.num_coords,
            'cnn_hidden_dims': self.cnn_hidden_dims,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'lstm_bidirectional': self.lstm_bidirectional,
            'num_classes': self.num_classes,
            'total_params': sum(p.numel() for p in self.parameters())
        }

def create_cnn_lstm_model(
    num_classes: int = NUM_CLASSES
) -> CNNLSTMModel:
    return CNNLSTMModel(
        num_joints=NUM_JOINTS,
        num_coords=NUM_COORDS,
        cnn_hidden_dims=CNN_HIDDEN_DIMS,
        cnn_kernel_sizes=CNN_KERNEL_SIZES,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_num_layers=LSTM_NUM_LAYERS,
        lstm_dropout=LSTM_DROPOUT,
        lstm_bidirectional=LSTM_BIDIRECTIONAL,
        fc_hidden_size=FC_HIDDEN_SIZE,
        num_classes=num_classes,
        dropout=DROPOUT
    )
