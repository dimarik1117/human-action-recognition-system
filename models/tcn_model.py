import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

INPUT_SIZE = 66
NUM_CHANNELS = [64, 128, 256]
KERNEL_SIZE = 5
DILATIONS = [1, 2, 4, 8]
NUM_BLOCKS = 4
DROPOUT = 0.3

FC_HIDDEN_SIZE = 128
NUM_CLASSES = 10

class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = DROPOUT
    ):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        self.crop = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.residual is not None:
            residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        diff_y = residual.size(2) - out.size(2)
        if diff_y > 0:
            residual = residual[:, :, :-diff_y]
        elif diff_y < 0:
            out = out[:, :, :residual.size(2)]
        
        out = out + residual
        out = F.relu(out)
        
        return out

class TCNModel(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        num_channels: List[int] = None,
        kernel_size: int = KERNEL_SIZE,
        dilations: List[int] = None,
        num_blocks: int = NUM_BLOCKS,
        num_classes: int = NUM_CLASSES,
        fc_hidden_size: int = FC_HIDDEN_SIZE,
        dropout: float = DROPOUT
    ):
        super(TCNModel, self).__init__()
        
        if num_channels is None:
            num_channels = NUM_CHANNELS
        if dilations is None:
            dilations = DILATIONS
        
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.fc_hidden_size = fc_hidden_size
        
        self.input_conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_channels[0],
            kernel_size=1
        )
        
        blocks = []
        in_channels = num_channels[0]
        
        for i in range(num_blocks):
            dilation = dilations[i % len(dilations)]
            out_channels = num_channels[min(i + 1, len(num_channels) - 1)]
            
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            in_channels = out_channels
        
        self.blocks = nn.Sequential(*blocks)
        
        tcn_output_size = in_channels
        
        if fc_hidden_size > 0:
            self.fc = nn.Sequential(
                nn.Linear(tcn_output_size, fc_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_hidden_size, num_classes)
            )
        else:
            self.fc = nn.Linear(tcn_output_size, num_classes)
        
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
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        x = x.transpose(1, 2)
        
        x = self.input_conv(x)
        
        x = self.blocks(x)
        
        x = x.mean(dim=2)
        
        if return_features:
            return x
        
        out = self.fc(x)
        
        return out
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        return probs, preds
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'model_type': 'TCN',
            'input_size': self.input_size,
            'num_channels': self.num_channels,
            'kernel_size': self.kernel_size,
            'num_blocks': self.num_blocks,
            'num_classes': self.num_classes,
            'fc_hidden_size': self.fc_hidden_size,
            'total_params': sum(p.numel() for p in self.parameters())
        }

def create_tcn_model(
    num_classes: int = NUM_CLASSES,
    num_channels: List[int] = None,
    kernel_size: int = KERNEL_SIZE,
    num_blocks: int = NUM_BLOCKS
) -> TCNModel:
    if num_channels is None:
        num_channels = NUM_CHANNELS
    
    return TCNModel(
        input_size=INPUT_SIZE,
        num_channels=num_channels,
        kernel_size=kernel_size,
        num_blocks=num_blocks,
        num_classes=num_classes,
        fc_hidden_size=FC_HIDDEN_SIZE,
        dropout=DROPOUT
    )
