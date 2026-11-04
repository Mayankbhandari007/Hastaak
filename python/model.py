import torch.nn as nn


class SignNet(nn.Module):
    """Small temporal classifier: 2 × Conv1d → BiGRU → mean pool → linear."""
    def __init__(self, num_classes: int, feat: int = 156, hidden: int = 192):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(0.3))
        self.gru = nn.GRU(hidden, 128, batch_first=True, bidirectional=True)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):                         # x: (batch, 32, 156)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(h.mean(1))