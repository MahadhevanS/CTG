"""1D-ResNet trivial baseline on raw FHR+UC (plan section 0.3)."""
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNet1D(nn.Module):
    """Input: (B, 3, T) [FHR, UC, missingness_mask]. Output: (B, 1) logit."""
    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 n_blocks_per_stage: int = 2, n_stages: int = 3, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )

        stages = []
        channels = base_channels
        for stage_i in range(n_stages):
            for _ in range(n_blocks_per_stage):
                stages.append(ResBlock1D(channels))
            if stage_i < n_stages - 1:
                next_channels = channels * 2
                stages.append(nn.Sequential(
                    nn.Conv1d(channels, next_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm1d(next_channels),
                    nn.ReLU(inplace=True),
                ))
                channels = next_channels
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stages(h)
        h = self.pool(h).squeeze(-1)
        return self.head(h)
