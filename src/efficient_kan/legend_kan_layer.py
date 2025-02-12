import torch
import torch.nn.functional as F
from typing import List
import math
import torch

class LegendreKANLayer(torch.nn.Module):

    def __init__(
        self,
        I: int,
        O: int,
        K: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
    ):
        super(LegendreKANLayer, self).__init__()
        self.I: int = I
        self.O: int = O
        self.K: int = K

        self.base_weight: torch.tensor = torch.nn.Parameter(torch.Tensor(O, I))
        self.spline_weight: torch.tensor = torch.nn.Parameter(torch.Tensor(O, I, G + K))
        self.spline_scaler: torch.tensor = torch.nn.Parameter(torch.Tensor(O, I))

        self.init_parameters()

# grid is of shape I x (G + 2 * K + 1)

def init_parameters(self):
    torch.nn.init.kaiming_uniform_(
        self.base_weight, a=math.sqrt(5) * self.scale_base
    )
    with torch.no_grad():
        noise = (
            (torch.rand(self.G + 1, self.I, self.O) - 1 / 2)
            * self.scale_noise
            / self.G
        )

        # fits the coefficients of the spline to some noise data
        coefs_from_spline_data: torch.tensor = self.coefs_from_curve_data(
            self.grid.T[self.K : -self.K],
            noise,
        )

        self.spline_weight.data.copy_(
            (self.scale_spline) * coefs_from_spline_data
        )

        # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
        torch.nn.init.kaiming_uniform_(
            self.spline_scaler, a=math.sqrt(5) * self.scale_spline
        )


#Dynamically assign the function to the class
LegendreKANLayer.init_parameters = init_parameters


def coefs_from_curve_data(self, x: torch.Tensor, y: torch.Tensor):
    r"""
    Compute the coefficients of the curve that interpolates the given points.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

    Returns:
        torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
    r"""
    assert x.dim() == 2 and x.size(1) == self.I
    assert y.size() == (x.size(0), self.I, self.O)

    # A vector or self.grid[K, -K]comees in as shape (2K, I)

    # y vector, or noise, comes in of shape
    # (G+1 x I x O) but gets changed to be of shape (I x G+1 x O)

    # of shape (B x I x G+K ) but of shpae (I x B x G+K) after transpose
    A = self.b_splines(x).transpose(
        0, 1
    )  # (in_features, batch_size, grid_size + spline_order)
    # gets changed to be of shape (I x G+1, O)
    B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
    solution = torch.linalg.lstsq(
        A, B
    ).solution  # (in_features, grid_size + spline_order, out_features)
    result = solution.permute(
        2, 0, 1
    )  # (out_features, in_features, grid_size + spline_order)

    assert result.size() == (
        self.O,
        self.I,
        self.G + self.K,
    )
    return result.contiguous()


#Dynamically assign the function to the class
LegendreKANLayer.coefs_from_curve_data = coefs_from_curve_data
@property
def scaled_spline_weight(self) -> torch.tensor:
    r"""Returns the scaled spline weight. In some of the other repositories, this is optional

    Returns:
        torch.tensor: the self.spline_weight tensor but scaled
    r"""
    return self.spline_weight * (self.spline_scaler.unsqueeze(-1))


#Dynamically assign the function to the class
LegendreKANLayer.scaled_spline_weight = scaled_spline_weight

def forward(self, x: torch.Tensor):
    assert x.dim() == 2 and x.size(1) == self.I

    base_output = F.linear(self.b(x), self.base_weight)
    b_spline_out: torch.tensor = self.b_splines(x)
    spline_output = F.linear(
        b_spline_out.view(x.size(0), -1),
        self.scaled_spline_weight.view(self.O, -1),
    )

    output : torch.tensor = base_output

    if self.use_splines:
        output = base_output + spline_output

    return output

#Dynamically assign the function to the class
LegendreKANLayer.forward = forward

# def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
#     r"""
#     Compute the regularization loss.

#     This is a dumb simulation of the original L1 regularization as stated in the
#     paper, since the original one requires computing absolutes and entropy from the
#     expanded (batch, in_features, out_features) intermediate tensor, which is hidden
#     behind the F.linear function if we want an memory efficient implementation.

#     The L1 regularization is now computed as mean absolute value of the spline
#     weights. The authors implementation also includes this term in addition to the
#     sample-based regularization.
#     r"""
#     l1_fake = self.spline_weight.abs().mean(-1)
#     regularization_loss_activation = l1_fake.sum()
#     p = l1_fake / regularization_loss_activation
#     regularization_loss_entropy = -torch.sum(p * p.log())
#     return (
#         regularize_activation * regularization_loss_activation
#         + regularize_entropy * regularization_loss_entropy
#         )

# #Dynamically assign the function to the class
# LegendreKANLayer.regularization_loss = regularization_loss

