import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

NUM_JOINTS = 33

GRAPH_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (0, 5), (0, 6),
    (6, 7), (7, 8),
    
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    
    (23, 25), (25, 27), (27, 29), (29, 31),
    
    (24, 26), (26, 28), (28, 30), (30, 32),
]

IN_CHANNELS = 2
NUM_CLASSES = 10
HIDDEN_CHANNELS = [64, 128, 256]
TEMPORAL_KERNEL_SIZE = 9
DROPOUT = 0.3

PARTITION_STRATEGY = 'spatial' # 'universal' или 'distance'

CENTER_INDICES = [23, 24, 11, 12]

class Graph:
    def __init__(
        self,
        num_joints: int = NUM_JOINTS,
        edges: List[Tuple[int, int]] = None,
        partition_strategy: str = PARTITION_STRATEGY,
        center_indices: List[int] = None
    ):
        if edges is None:
            edges = GRAPH_EDGES
        if center_indices is None:
            center_indices = CENTER_INDICES
        
        self.num_joints = num_joints
        self.edges = edges
        self.partition_strategy = partition_strategy
        self.center_indices = center_indices
        
        self.neighbors = [[] for _ in range(num_joints)]
        for u, v in edges:
            self.neighbors[u].append(v)
            self.neighbors[v].append(u)
        
        for i in range(num_joints):
            self.neighbors[i].append(i)
            self.neighbors[i] = list(set(self.neighbors[i]))
        
        self.A = self._build_adjacency_matrix()
    
    def _get_hop_distance(self) -> np.ndarray:
        num_joints = self.num_joints
        dist = np.full((num_joints, num_joints), num_joints, dtype=np.float32)
        
        for i in range(num_joints):
            dist[i, i] = 0
        
        for i in range(num_joints):
            queue = [i]
            visited = set([i])
            depth = 0
            while queue:
                next_queue = []
                for u in queue:
                    for v in self.neighbors[u]:
                        if v not in visited:
                            visited.add(v)
                            next_queue.append(v)
                            dist[i, v] = depth + 1
                queue = next_queue
                depth += 1
        
        return dist
    
    def _build_adjacency_matrix(self) -> np.ndarray:
        if self.partition_strategy == 'universal':
            A = np.zeros((1, self.num_joints, self.num_joints), dtype=np.float32)
            for i in range(self.num_joints):
                for j in self.neighbors[i]:
                    A[0, i, j] = 1.0 / len(self.neighbors[i])
        
        elif self.partition_strategy == 'distance':
            A = np.zeros((2, self.num_joints, self.num_joints), dtype=np.float32)
            for i in range(self.num_joints):
                for j in self.neighbors[i]:
                    if j == i:
                        A[0, i, j] = 1.0
                    else:
                        A[1, i, j] = 1.0
        
        elif self.partition_strategy == 'spatial':
            A = np.zeros((3, self.num_joints, self.num_joints), dtype=np.float32)
            for i in range(self.num_joints):
                for j in self.neighbors[i]:
                    if j == i:
                        A[0, i, j] = 1.0
                    else:
                        A[1, i, j] = 1.0
        
        else:
            raise ValueError(f"Неизвестная стратегия разбиения: {self.partition_strategy}")
        
        for k in range(A.shape[0]):
            for i in range(self.num_joints):
                row_sum = np.sum(A[k, i, :])
                if row_sum > 0:
                    A[k, i, :] /= row_sum
        
        return A
    
    def get_adjacency_matrix(self) -> torch.Tensor:
        return torch.from_numpy(self.A).float()

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, graph: Graph):
        super(ConvTemporalGraphical, self).__init__()
        
        self.graph = graph
        self.A = graph.get_adjacency_matrix()
        self.num_groups = self.A.shape[0]
        
        self.conv = nn.Conv2d(
            in_channels * self.num_groups,
            out_channels * self.num_groups,
            kernel_size=(1, 1),
            groups=self.num_groups
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        A = self.A.to(x.device)
        
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        
        out = torch.einsum('ntvc, gvw -> ntgwc', x_perm, A)
        
        out = out.permute(0, 4, 1, 2, 3).contiguous()
        out = out.view(N, C * self.num_groups, T, V)
        
        out = self.conv(out)
        
        return out


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        graph: Graph,
        temporal_kernel_size: int = TEMPORAL_KERNEL_SIZE,
        dropout: float = DROPOUT
    ):
        super(STGCNBlock, self).__init__()
        
        self.num_groups = graph.A.shape[0]
        
        self.spatial = ConvTemporalGraphical(in_channels, out_channels, graph)
        
        temporal_in_channels = out_channels * self.num_groups
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(temporal_in_channels),
            nn.ReLU(),
            nn.Conv2d(
                temporal_in_channels,
                temporal_in_channels,
                kernel_size=(temporal_kernel_size, 1),
                padding=((temporal_kernel_size - 1) // 2, 0)
            ),
            nn.BatchNorm2d(temporal_in_channels),
            nn.Dropout(dropout)
        )
        
        self.reduce_channels = nn.Conv2d(temporal_in_channels, out_channels, kernel_size=1)
        
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.residual is not None:
            residual = self.residual(x)
        
        out = self.spatial(x)
        out = self.temporal(out)
        out = self.reduce_channels(out)
        
        out = out + residual
        return F.relu(out)

class STGCNModel(nn.Module):
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        hidden_channels: List[int] = None,
        num_joints: int = NUM_JOINTS,
        edges: List[Tuple[int, int]] = None,
        temporal_kernel_size: int = TEMPORAL_KERNEL_SIZE,
        dropout: float = DROPOUT,
        partition_strategy: str = PARTITION_STRATEGY
    ):
        super(STGCNModel, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = HIDDEN_CHANNELS
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.num_joints = num_joints
        
        self.graph = Graph(
            num_joints=num_joints,
            edges=edges,
            partition_strategy=partition_strategy
        )
        
        self.input_conv = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=1)
        
        self.input_bn = nn.BatchNorm2d(hidden_channels[0])
        
        self.stgcn_blocks = nn.ModuleList()
        
        for i, (in_ch, out_ch) in enumerate(zip([hidden_channels[0]] + hidden_channels[:-1], hidden_channels)):
            self.stgcn_blocks.append(
                STGCNBlock(in_ch, out_ch, self.graph, temporal_kernel_size, dropout)
            )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels[-1], num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
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
        batch_size, seq_len, num_joints, num_coords = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        for block in self.stgcn_blocks:
            x = block(x)
        
        x = self.global_avg_pool(x)
        x = x.view(batch_size, -1)
        
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
            'model_type': 'ST-GCN',
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'hidden_channels': self.hidden_channels,
            'num_joints': self.num_joints,
            'temporal_kernel_size': TEMPORAL_KERNEL_SIZE,
            'partition_strategy': PARTITION_STRATEGY,
            'total_params': sum(p.numel() for p in self.parameters())
        }
    
    def get_adjacency_matrix(self) -> np.ndarray:
        return self.graph.A

def create_stgcn_model(
    num_classes: int = NUM_CLASSES,
    in_channels: int = IN_CHANNELS,
    hidden_channels: List[int] = None
) -> STGCNModel:
    if hidden_channels is None:
        hidden_channels = HIDDEN_CHANNELS
    
    return STGCNModel(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        num_joints=NUM_JOINTS,
        edges=GRAPH_EDGES,
        temporal_kernel_size=TEMPORAL_KERNEL_SIZE,
        dropout=DROPOUT,
        partition_strategy=PARTITION_STRATEGY
    )
