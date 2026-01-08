from typing import Dict

import torch
from torch import nn

from module.model.encoders.base import ModalityName


class SimpleConcat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        # each output should be B, T, C

        return torch.concat(
            [x[modality_name] for modality_name in list(ModalityName) if modality_name in x.keys()], dim=-1
        )
