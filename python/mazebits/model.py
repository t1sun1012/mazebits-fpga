"""A compact spatial CNN policy suitable for a later fixed-point implementation."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 16
    depth: int = 8

    def __post_init__(self):
        if type(self.channels) is not int or self.channels < 1:
            raise ValueError("channels must be a positive integer")
        if type(self.depth) is not int or self.depth < 1:
            raise ValueError("depth must be a positive integer")


class PolicyCNN(nn.Module):
    """3x8x8 binary state -> four raw action scores, without action masking.

    Each 3x3 convolution is followed by ReLU. The first convolution sees a
    virtual obstacle ring outside the maze; hidden feature padding is zero.
    A 1x1 head produces four scores per cell. The agent plane selects a cell.
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.convs = nn.ModuleList([
            nn.Conv2d(3 if i == 0 else config.channels, config.channels, 3,
                      padding=0 if i == 0 else 1)
            for i in range(config.depth)
        ])
        self.head = nn.Conv2d(config.channels, 4, 1)
        for layer in self.convs:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def action_field(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (3, 8, 8):
            raise ValueError("inputs must have shape (N, 3, 8, 8)")
        padded = torch.cat((
            F.pad(inputs[:, :1], (1, 1, 1, 1), value=1.0),
            F.pad(inputs[:, 1:], (1, 1, 1, 1), value=0.0),
        ), dim=1)
        hidden = F.relu(self.convs[0](padded))
        for conv in self.convs[1:]:
            hidden = F.relu(conv(hidden))
        return self.head(hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        field = self.action_field(inputs).flatten(2)
        index = inputs[:, 1].flatten(1).argmax(dim=1)
        return field.gather(2, index[:, None, None].expand(-1, 4, 1)).squeeze(2)

    def description(self) -> dict:
        return {
            "architecture": "spatial-policy-cnn-v1",
            "config": asdict(self.config),
            "input_shape": [3, 8, 8],
            "input_channels": ["walls", "agent", "goal"],
            "action_order": ["UP", "RIGHT", "DOWN", "LEFT"],
            "parameters": sum(p.numel() for p in self.parameters()),
            "weights": sum(m.weight.numel() for m in [*self.convs, self.head]),
            "biases": sum(m.bias.numel() for m in [*self.convs, self.head]),
            "dense_macs_per_decision": sum(
                m.weight.numel() * 64 for m in [*self.convs, self.head]
            ),
            "receptive_field": 1 + 2 * self.config.depth,
            "first_layer_padding": "wall channel 1; agent and goal channels 0",
            "hidden_padding": "zero",
            "readout": "four action-head scores at the one-hot agent cell",
        }


def optimal_set_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Negative log probability mass assigned to the set of optimal actions."""
    if logits.ndim != 2 or logits.shape[1] != 4 or targets.shape != logits.shape:
        raise ValueError("logits and targets must both have shape (N, 4)")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("targets must be binary optimal-action indicators")
    mask = targets.bool()
    if not mask.any(dim=1).all():
        raise ValueError("every supervised state needs at least one optimal action")
    return (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=1)
    ).mean()


def augment_batch(
    inputs: torch.Tensor, targets: torch.Tensor, turns: int, reflect: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a D4 symmetry, preserving action IDs (torch rotations are CCW)."""
    if type(turns) is not int or turns not in range(4):
        raise ValueError("turns must be one of 0, 1, 2, 3")
    augmented = torch.rot90(inputs, turns, dims=(-2, -1))
    labels = torch.roll(targets, shifts=-turns, dims=1)
    if reflect:
        augmented = augmented.flip(-1)
        labels = labels[:, [0, 3, 2, 1]]
    return augmented, labels
