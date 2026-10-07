"""Assignment architectures. All autoencoder hidden activations are logistic sigmoid."""
from __future__ import annotations

import torch
from torch import nn

ARCHITECTURES = {"64": (64,), "128": (128,), "128_64": (128, 64)}


class Autoencoder(nn.Module):
    def __init__(self, bottleneck: int, deep: bool = False):
        super().__init__()
        if deep:
            self.encoder = nn.Sequential(nn.Linear(784, 400), nn.Sigmoid(),
                                         nn.Linear(400, bottleneck), nn.Sigmoid())
            self.decoder = nn.Sequential(nn.Linear(bottleneck, 400), nn.Sigmoid(),
                                         nn.Linear(400, 784), nn.Sigmoid())
        else:
            self.encoder = nn.Sequential(nn.Linear(784, bottleneck), nn.Sigmoid())
            self.decoder = nn.Sequential(nn.Linear(bottleneck, 784), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...], classes: int = 5):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def corrupt(x: torch.Tensor, probability: float, generator=None):
    """Independent Bernoulli masking noise: selected pixels are set to zero."""
    if probability == 0:
        return x
    keep = torch.rand(x.shape, generator=generator, device=x.device) >= probability
    return x * keep
