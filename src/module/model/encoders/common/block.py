import torch
from torch import nn


class PermuteBlock:
    def __init__(self, dimension_list: list[int]):
        self._dimension_list = dimension_list

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self._dimension_list)


class SimpleCNN1DBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(in_features)
        self.conv1 = nn.Conv1d(in_features, out_features, stride=1, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

        self.ln2 = nn.LayerNorm(out_features)
        self.conv2 = nn.Conv1d(out_features, out_features, stride=2, kernel_size=3, padding=1)
        self.act2 = nn.GELU()
        self.pooling = nn.MaxPool1d(2)
        self.permute = PermuteBlock(dimension_list=[0, 2, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.permute(self.ln1(x))  # B, T, C -> B, C, T
        out = out + self.act1(self.conv1(out))
        out = self.permute(out)  # B, C, T -> B, T, C

        out = self.permute(self.ln2(out))
        out = self.pooling(out) + self.act2(self.conv2(out))
        out = self.permute(out)  # B, C, T -> B, T, C
        return out
