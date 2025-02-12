import torch
from typing import List
import sys

from kan_linear import KANLinear

################################
# File for defining what a KAN is
################################

class LegendreKAN(torch.nn.Module):
    def __init__(
        self,
        layers : List[int],
        K: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        b : torch.nn.Module =torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range=[-1, 1],
    ):
        super(LegendreKAN, self).__init__()
        self.K: int = K

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers, layers[1:]):
            self.layers.append(
                LegendreKANLayer(
                    in_features,
                    out_features,
                    K=K,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    b=b,
                )
            )


def forward(self, x: torch.Tensor, update_grid=False):
    for layer in self.layers:
        x = layer(x)
    return x

#Dynamically assign the function to the class
LegendreKAN.forward = forward


def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
    return sum(
        layer.regularization_loss(regularize_activation, regularize_entropy)
        for layer in self.layers
    )

#Dynamically assign the function to the class
LegendreKAN.regularization_loss = regularization_loss