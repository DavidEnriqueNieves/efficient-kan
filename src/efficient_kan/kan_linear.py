#| **Debugging for Jupyter Notebook*

import platform
import sys  

print(sys.version)  
print(sys.version_info) 

print(f"Python Version: {sys.version}")  
print(f"Version Info: {sys.version_info}")  
print(f"Platform Version: {platform.python_version()}") 

# ------------------------------------------
#| For the purpose of initializing Plotly for the later plots

from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *

init_notebook_mode(connected=True)         # initiate notebook for offline plot

# ------------------------------------------

import shutil # Check path to Python executable  

python_exe = shutil.which("python")  

print(f"Python executable path: {python_exe}") # e.g. Python executable path: /usr/bin/python3  

# ------------------------------------------
import torch
import torch.nn.functional as F
from typing import List
import math
import torch

class KANLinear(torch.nn.Module):

    def __init__(
        self,
        I: int,
        O: int,
        G: int = 5,
        K: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        b=torch.nn.SiLU,
        use_splines : bool = True,
        grid_eps: float = 0.02,
        grid_range: List[int] = [-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.I: int = I
        self.O: int = O
        self.G: int = G
        self.K: int = K
        self.use_splines : bool = use_splines
        self.b: torch.nn.Module = b()

        h: float = (grid_range[1] - grid_range[0]) / G
        
        # grid is of shape G _+ 2K + 1 representing G + 2K points
        # the additional 2K points are needed for the boundary points
        # splines of order K are devined over G intervals as needing 
        grid: torch.tensor = (
            (torch.arange(-K, G + K + 1) * h + grid_range[0]).expand(I, -1).contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight: torch.tensor = torch.nn.Parameter(torch.Tensor(O, I))
        self.spline_weight: torch.tensor = torch.nn.Parameter(torch.Tensor(O, I, G + K))
        self.spline_scaler: torch.tensor = torch.nn.Parameter(torch.Tensor(O, I))

        self.scale_noise: float = scale_noise
        self.scale_base: float = scale_base
        self.scale_spline: float = scale_spline
        self.grid_eps: float = grid_eps

        self.init_parameters()

# grid is of shape I x (G + 2 * K + 1)

def init_parameters(self):
    r"""
    Uses Kaiming He initialization,
    a technique mainly used for initializing the weights of ReLU neural networks

    It initializes the weights such that the variance of the weights is the same across the network.

    The weights are initialized from a normal distribution $\mathcal{N}(0, \frac{2}{n}$

    It is related to the Xavier intialization, which is more suited for activation functions like sigmoid or tanh functions.

    For the spline fitting, the `noise` variable instantiation generates tensor
    of shape $(G+1 \times I \times O)$.  
    
    By default,
    [`torch.rand()`](https://pytorch.org/docs/stable/generated/torch.rand.html)
    generates values from $[0, 1)$ with a uniform distribution, so the 1/2
    centers the distribution. From there, a `scale_noise : float` factor is
    applied and the entire thing is divided by $G$.
    r"""
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
KANLinear.init_parameters = init_parameters


def b_splines(self, x: torch.Tensor):
    r"""
    Used for evaluating spline curves in B-spline form.
    r"""
    r"""
    Compute the B-spline bases for the given input tensor using the de Boor algorithm
    (Equivalent to coef2curve in the original repo)

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).

    Returns:
        torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
    r"""
    assert x.dim() == 2 and x.size(1) == self.I

    grid: torch.Tensor = (
        self.grid
    )  # (in_features, grid_size + 2 * spline_order + 1)
    x = x.unsqueeze(-1)

    # uses the de Boor algorithm
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for k in range(1, self.K + 1):
        bases = (
            (x - grid[:, : -(k + 1)])
            / (grid[:, k:-1] - grid[:, : -(k + 1)])
            * bases[:, :, :-1]
        ) + (
            (grid[:, k + 1 :] - x)
            / (grid[:, k + 1 :] - grid[:, 1:(-k)])
            * bases[:, :, 1:]
        )

    assert bases.size() == (
        x.size(0),
        self.I,
        self.G + self.K,
    )
    return bases.contiguous()

#Dynamically assign the function to the class
KANLinear.b_splines = b_splines


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
KANLinear.coefs_from_curve_data = coefs_from_curve_data
@property
def scaled_spline_weight(self) -> torch.tensor:
    r"""Returns the scaled spline weight. In some of the other repositories, this is optional

    Returns:
        torch.tensor: the self.spline_weight tensor but scaled
    r"""
    return self.spline_weight * (self.spline_scaler.unsqueeze(-1))


#Dynamically assign the function to the class
KANLinear.scaled_spline_weight = scaled_spline_weight

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
KANLinear.forward = forward

@torch.no_grad()
def update_grid(self, x: torch.Tensor, margin=0.01):
    r"""Update grid

    Recall: the grid is of shape: $\mathbb{R}^{(G + 2K + 1) \times I}$

    Args:
        x (torch.Tensor): input tensor to base the update off of
        margin (float, optional): Margin for the uniform extension of the grid. Defaults to 0.01.
    r"""
    assert x.dim() == 2 and x.size(1) == self.I
    batch = x.size(0)

    splines : torch.tensor = self.b_splines(x)  # (batch, in, coeff)
    splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
    orig_coeff : torch.tensor = self.scaled_spline_weight  # (out, in, coeff)
    orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
    unreduced_spline_output : torch.tensor = torch.bmm(splines, orig_coeff)  # (in, batch, out)
    unreduced_spline_output = unreduced_spline_output.permute(
        1, 0, 2
    )  # (batch, in, out)

    # sort each channel individually to collect data distribution
    x_sorted : torch.tensor = torch.sort(x, dim=0)[0]
    # selects G samples from x_sorted
    grid_adaptive : torch.tensor  = x_sorted[
        torch.linspace(0, batch - 1, self.G + 1, dtype=torch.int64, device=x.device)
    ]

    # take the highest and lowest x values respectively, add a margin, and divide by G
    uniform_step : torch.tensor = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.G
    # take a vector of shape (G + 1 x 1) and 
    #
    # e.g. a tensor of this form
    # tensor([[0.],
    #     [1.],
    #     [2.],
    #     [...],
    #     [G]])
    # then, multiply it by the uniform step and add by the lower bound to frame
    # it within the bounds of the highest and lowest x values plus a margin so far
    grid_uniform : torch.tensor = (
        torch.arange(self.G + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
        * uniform_step
        + x_sorted[0]
        - margin
    )

    grid : torch.tensor = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
    grid = torch.concatenate(
        [
            grid[:1]
            - uniform_step
            * torch.arange(self.K, 0, -1, device=x.device).unsqueeze(1),
            grid,
            grid[-1:]
            + uniform_step
            * torch.arange(1, self.K + 1, device=x.device).unsqueeze(1),
        ],
        dim=0,
    )

    # adds 2k steps for the purpose of evaluating spline at boundary points
    # grid is now of shape self.G + 2 * self.K + 1
    self.grid.copy_(grid.T)

    # fits the curve once again from the updated grid using the same old data points as before
    self.spline_weight.data.copy_(
        self.coefs_from_curve_data(x, unreduced_spline_output)
    )

#Dynamically assign the function to the class
KANLinear.update_grid = update_grid

def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
    r"""
    Compute the regularization loss.

    This is a dumb simulation of the original L1 regularization as stated in the
    paper, since the original one requires computing absolutes and entropy from the
    expanded (batch, in_features, out_features) intermediate tensor, which is hidden
    behind the F.linear function if we want an memory efficient implementation.

    The L1 regularization is now computed as mean absolute value of the spline
    weights. The authors implementation also includes this term in addition to the
    sample-based regularization.
    r"""
    l1_fake = self.spline_weight.abs().mean(-1)
    regularization_loss_activation = l1_fake.sum()
    p = l1_fake / regularization_loss_activation
    regularization_loss_entropy = -torch.sum(p * p.log())
    return (
        regularize_activation * regularization_loss_activation
        + regularize_entropy * regularization_loss_entropy
        )

#Dynamically assign the function to the class
KANLinear.regularization_loss = regularization_loss
