"""Network definition shared by offline training and local inference."""

from __future__ import annotations

import torch
from torch import nn


class TerrainCNN(nn.Module):
    def __init__(self, output_classes: int = 1):
        super().__init__()
        self.output_classes = output_classes
        blocks = []
        channels = 9
        for output in (16, 32, 64):
            blocks.extend([
                nn.Conv2d(channels, output, 3, padding=1), nn.BatchNorm2d(output), nn.ReLU(),
                nn.Conv2d(output, output, 3, padding=1), nn.BatchNorm2d(output), nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            channels = output
        self.encoder = nn.Sequential(*blocks)
        self.pool_mean = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.tabular = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(160, 64), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(64, output_classes))

    def forward(self, patch: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        features = self.encoder(patch)
        joined = torch.cat((self.pool_mean(features).flatten(1),
                            self.pool_max(features).flatten(1), self.tabular(values)), dim=1)
        logits = self.head(joined)
        return logits.squeeze(1) if self.output_classes == 1 else logits
