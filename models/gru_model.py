import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

INPUT_SIZE = 66
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = False

FC_HIDDEN_SIZE = 64
NUM_CLASSES = 10

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4

class GRUModel(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
        bidirectional: bool = BIDIRECTIONAL,
        fc_hidden_size: int = FC_HIDDEN_SIZE
    ):
        super(GRUModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.fc_hidden_size = fc_hidden_size
        
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        
        gru_output_size = hidden_size * self.num_directions
        
        if fc_hidden_size > 0:
            self.fc = nn.Sequential(
                nn.Linear(gru_output_size, fc_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden_size, num_classes)
            )
        else:
            self.fc = nn.Linear(gru_output_size, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        gru_out, hidden = self.gru(x)
        
        if self.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
            features = hidden_concat
        else:
            features = hidden[-1, :, :]
        
        features = self.dropout(features)
        
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
            'model_type': 'GRU',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout.p,
            'bidirectional': self.bidirectional,
            'fc_hidden_size': self.fc_hidden_size,
            'total_params': sum(p.numel() for p in self.parameters())
        }

def create_gru_model(
    num_classes: int = NUM_CLASSES,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT
) -> GRUModel:
    return GRUModel(
        input_size=INPUT_SIZE,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=BIDIRECTIONAL,
        fc_hidden_size=FC_HIDDEN_SIZE
    )
