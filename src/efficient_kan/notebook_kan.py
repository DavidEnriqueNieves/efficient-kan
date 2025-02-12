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
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
import sys
sys.path.append("/home/davidn/PhXD/KAN/efficient-kan/src/efficient_kan/")
from utils import CommonUtils
from pathlib import Path

#| # What is this notebook for?
#| This is a modified version of the `kan.py` file from the original
#| `efficient-kan` repository. It's a work in progress

#| ## KAN & KANLinear Class definitions
#| Some important definitions include
#| 
#| $I \in \mathbb{N}$ , the input size of the layer, represented by `self.I` in the
#| code
#| 
#| $O \in \mathbb{N}$, the output size of the layer, represented by `self.O` in the
#| code
#| 
#| $G \in \mathbb{N}$, the number of grid intervals used in the spline, represented
#| by `self.G` in the code
#| 
#| $K \in \mathbb{N}$, the order of the spline to be used for activations,
#| represented by `self.K` in the code
#| 
#| $\text{grid} \in \mathbb{R}^{(G + 2K + 1) \times I}$, the grid on which to evaluate
#| the splines on, represented by `self.grid` in the code. It Normally, the grid would require G + K + 1 points, but
#| the extra K comes from evaluating the boundary points of the spline.
#| 
#| $b : \mathbb{R} \mapsto \mathbb{R}$ will be a function applied elementwise to any matrix
#| that serves as its input, represented by `self.b : torch.nn.Module` in the
#| code and having $\text{SiLU}$ as its default value
#| 
#| Note that the $b$ corresponds to the basis function mentioned in equation
#| (2.10) of the paper, where the spline function is equal to:
#| 
#| $$ \phi(x) = w(b(x) + \text{spline}(x)) $$
#| 
#| ... with $b(x) = \text{silu}(x)$ in the paper.
#| 
#| $h \in \mathbb{R}$, the grid interval used for initializing the grid, represented by
#| `h : float` in the code
#| 
#| #### Learnable parameters
#| 
#| $W_{\text{base}}$, the weight used to scale the base activations, of shape $O \times I$
#|
#|
#| $W_{\text{spline}}$, the weight used to scale the spline activations, of shape $O \times I \times G+K$
#| 
#| `spline_scaler`, the weight used to scale the spline weights across the $G + K$ dimension
#| 
#| Note that the grid updates and is by proxy *learnable*, but it is highly dependent on the input.
#| 
#| `self.scale_noise`, `self.scale_base`, `self.scale_spline`, and
#| 
#| `self.gird_eps` are used later in the `self.init_parameters()` function call

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

#| #### Weight Initialization
#| 
#| Naturally, initializing the neurons to be the same value is redundant and
#| inefficient, since you'll have neurons that take on the same values and do the
#| same thing even after backpropagation. It's only natural that weights are
#| therefore intialized randomly. This has an effect on the convergence of the
#| network. However, some weight initialization schemes are more effective than others.
#| 
#| The rational behind the scaling of the layers is to account for the variance of the previous
#| layer and prevent the vanishing or explosion of the gradient in deeper networks.
#| 
#| Ideally, as reasoned in Gloriot & Bengio (2010), it is the case that
#| 
#| The intuition then goes that having the updates spread out in the same way would
#| mean that the layers are learning at the same rythm, thus mitigating the effects
#| of vanishing or exploding gradients.
#| (obtained from [here](https://datascience.stackexchange.com/questions/82917/why-do-we-want-the-variance-of-the-layers-to-remain-the-same-throughout-a-deep-n))
#| 
#| Thus, if that is the case, then:
#| 
#| $$
#| \text{Layer 1}\rightarrow \frac{\partial Cost}{\partial W^i}= \frac{\partial Cost}{\partial s^i}\frac{\partial s^i}{\partial W^i}=\frac{\partial Cost}{\partial s^i} (z^{i-1})^T \\
#| \text{Layer 2}\rightarrow \frac{\partial Cost}{\partial W^{i+1}}= \frac{\partial Cost}{\partial s^{i+1}}\frac{\partial s^{i+1}}{\partial W^{i+1}}=\frac{\partial Cost}{\partial s^{i+1}} (z^{i})^T
#| $$
#| 
#| Thus, considering that if:
#| $$
#| Var\left(\frac{\partial Cost}{\partial s^i}\right) = Var\left(\frac{\partial Cost}{\partial s^{i+1}}\right) \,\,\,\,\,\leftrightarrow\,\,\,\,\, Var(z^i)=Var(z^{i-1})
#| $$
#| 
#| then, this would mean:
#| 
#| $$
#| Var\left(\frac{\partial Cost}{\partial W^i} \right) = Var\left(\frac{\partial Cost}{\partial W^{i+1}} \right)
#| $$
#| 
#| Hence why initializing the weights to have the same variance is important.
#| 
#| #### Xavier Initialization
#| 
#| In the original KAN paper, they used Xavier initialization.
#| The initialization technique is listed as the second implementation detail of the
#| paper.
#| 
#| > Initialization scales. Each activation function is initialized to have
#| spline(x) ≈ 0 2. w is initialized according to the Xavier initialization, which
#| has been used to initialize linear layers in MLPs.
#| 
#| 
#| A very abbreviated rundown of the two goes as follows:
#| 
#| Xavier initialization, scales the weights proportional to the number of inputs to the layer.
#| 
#| This means that assuming a bias of zero, you initialize your weights to be 
#| 
#| $$
#| W_{i,j}^{(l)} \sim N(\mu = 0, \sigma^2 = \frac{1}{\sqrt{m^{(l-1)}}})
#| $$
#| ... where $m$ is the number of input units to the next layer.
#| 
#| Note that sometimes $m^{l-1} + m^{l}$ is used.
#| 
#| However, one assumption of Xavier intialization was a derivative of 1 for the
#| activation function and mean of 0 for activations.
#| 
#| [Helpful video](https://www.youtube.com/watch?v=ScWTYHQra5E)
#| 
#| #### Kaiming He Intialization
#| 
#| The difference between Kaiming He initialization and Xavier intialization is
#| that Kaiming He takes into account the use of ReLU units. The approach is fairly
#| similar to Xavier intialization. The end result is a scaling factor that looks
#| like the following
#| 
#| 
#| $$
#| W_{i,j}^{(l)} \sim N(\mu = 0, \sigma^2 = \sqrt{\frac{2}{m^{(l-1)}}})
#| $$
#| 
#| #### Fitting the spline weight
#| 
#| The weights of the splines are not initialized using Kaiming He initialization. Instead, random data 
#| 
#| 

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


#| This function makes use of the Cox de Boor algorithm, which in turn uses the de Boor recursive formula
#| 
#| $$
#| B_{i,0}(t) = \begin{cases} 
#|       1 & \text{if} & t_i \leq t < t_{i+1} \\
#|       0 & \text{otherwise}
#|    \end{cases}\\
#|    B_{i,p}(t) = \frac{t-t_i}{t_{i+p} - t_i}B_{i,p-1}(t)+\frac{t_{i+1}-t}{t_{i+p}-t_i}B_{i+1, p-1}(t)
#| $$
#| 
#| ... where $t_i$ is the $i$th entry of the knot vector in the spline. This just
#| means the $i$ th input at which your $i$ th control point will be evaluated at.
#| 
#| Note that since the `grid : torch.tensor` object is handling the grid values at
#| different inputs (there is an independent grid for each input), as well as for
#| different piecewise segments of the spline. Hence, the grid only has two
#| dimensions and will always be indexed with just two indices in the code.
#| 
#| The `bases` variable on the other hand, will have three indices. This is because
#| of the $B_{i,p}(t)$ function mentioned above taking in two indices $(i, p)$, and the
#| argument of $t$. Note that $p$ is the order of the B-spline polynomial being
#| evaluated, $i$ is the index corresponding to which segment will be evaluated
#| against, and $t$ is the actual value to use in the evaluation.
#| 
#| TODO: finish analyzing this
#| In the case of the `bases` variable, it seems that the first index is the batch
#| dimension corresponding to $t$ in the 

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

#| The solution below computes a solution to the linear system
#| 
#| $$
#|     \underset{X \in \mathbb{K}}{\text{argmin}} \ \ ||AX - B||_{F}
#| $$
#| 
#| ... where in this case, $A$ is taken to be the input tensor of shape $(B
#| \times I)$, $B$ is the input tensor of shape $B \times I \times O$, and $|| \cdot ||_F$ is the Frobenius norm.
#| 
#| It makes use of the [`torch.linalg.lstsq()`](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) function. 
#| 
#| In other words, it simply obtains the coefficients of the splines that best correspond to the data provided through $B$.

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

#| Recall that the `self.spline_weight : torch.tensor` member is of shape $O \times I \times (G + K)$
#| Similarly, `self.spline_scaler : torch.tensor` is of shape $O \times I$. 
#| 
#| What the operation below does is turn `spline_scaler` into an $O \times I \times
#| 1$ tensor and then broadcast the multiplication across the $(G + K)$ axis. What
#| this means is that `self.spline_weight` is responsible for scaling each of the
#| input splines will be scaled by a certain amount depending on which node it is
#| going to. Essentially, it applies a weight across the $(G + K)$ dimension,
#| indicating that is is weighing the *whole* spline, globally.

@property
def scaled_spline_weight(self) -> torch.tensor:
    r"""Returns the scaled spline weight. In some of the other repositories, this is optional

    Returns:
        torch.tensor: the self.spline_weight tensor but scaled
    r"""
    return self.spline_weight * (self.spline_scaler.unsqueeze(-1))


#Dynamically assign the function to the class
KANLinear.scaled_spline_weight = scaled_spline_weight

#| For this function, the output is remarkably simple compared to the original implementation.
#| It has two components: a `base_output` object, and a `spline_output` object. 
#| 
#| The input $X \in \mathbb{R}^{B \times I}$ first gets evaluated element-wise with the basis function $b(\cdot)$. There is a `base_out` object, which is a tensor that is a result of multiplying $b(X) \in \mathbb{R}^{B \times I}$ with $W_{\text{base}}^{\intercal} \in \mathbb{R}^{I \times O}$ through the action of the python function [`F.linear()`](https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html). 
#| 
#| Next, the result of the B-spline evaluation with the points of $X$ is calculated using the function `self.b_splines()`.  This is then scaled by `scaled_spline_weight`, which is a weight for the splines which combines both the regular weight $W_{spline} \in \mathbb{R}^{O \times I \times (G+K)}$ and a scaler known as `self.spline_scaler` which is in $\mathbb{R} ^{O \times I}$  and has the purpose of broadcasting a weight for every input spline of every node in this layer. In other words, it scales all parts of each input spline of each node *equally* as opposed to scaling certain parts of the spline differently. Of course, this *equal* weighing will still vary across the inputs for every node in this layer.
#| 
#| Finally, after applying the scaling to the output of `b_splines()` , the spline component and the base component of the output are added together as in equation (2.10) of the paper.
#| 

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

#| This is noted in the paper as being one of the key implementation
#| details to making 
#| KANs optimizable. The grid update step is one of these implementation details.
#| 
#| > Update of spline grids. We update each grid on the fly according to its input
#| activations, to address the issue that splines are defined on bounded regions
#| but activation values can evolve out of the fixed region during training 3.
#| 
#| NOTE that this means that the splines will still evaluate, but it's likely that
#| the result will not be favorable due to the coefficients of the splines not
#| necessarily accounting for that input region.
#| 
#| 
#| Noting that $X \in \mathbb{R}^{B \times I}$, what this step does is that it sorts the
#| $X$ input values along the $B$ dimension, and does two things:
#| 
#| it creates a `grid_adaptive` tensor and a `grid_uniform` tensor
#| 
#| With the grid_adaptive tensor, it simply takes $G + 1$ samples from `x_sorted`
#| With `grid_uniform`, it creates evenly spaced $G+1$ samples from the highest and
#| lowest values of $X$ and populates the array using those esmaples
#| 
#| Afterwards, `grid` is reinitialized as a convex combination of `grid_uniform` and
#| `grid_adaptive` using `self.grid_eps : float` as the factor. Note that by
#| default, `self.grid_eps` is fairly small, so `grid` will mostly take on the
#| values of `grid_adaptive`, i.e. non-evenly spaced samples from the original
#| sorted $X$ along the $B$ dimension.
#| 
#| Finally, `grid` is added two extensions at  the start and at the end, each a
#| `torch.tensor` of length $K$ for the purpose of evaluating the spline at the
#| boundary points.

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

#| In Section 2.5.1 of the paper, they define L1 regularization and later on,
#| entropy regularization.
#| 
#| For the definition of the L1 norm in (2.17), they use 
#| 
#| $$
#| |\phi|_1 = \frac{1}{N_p} \sum_{s=1}^{N_p} \left| \phi(x^{(s)}) \right|
#| $$
#| 
#| Hence, for the entire layer, the L1 norm (2.18) is defined as
#| 
#| $$
#| |\Phi|_1  = \sum_{i=1}^{n_{in}}\sum_{j=1}^{n_{out}} |\phi_{i,j}|_1
#| $$
#| 
#| Furthermore, the entropy (2.19) is defined to be
#| 
#| $$
#| S(\Phi) = -\sum_{i=1}^{n_{in}} \sum_{j=1}^{n_{out}}  \frac{|\phi_{i,j}|_1}{|\Phi|_1} \text{log} \left( \frac{|\phi_{i,j}|_1}{|\Phi|_1}  \right)
#| $$
#| 
#| See equation (2.20)
#| $$
#| \ell_{\text{total}} = \ell_{\text{pred}} + \lambda \left( \mu_1 \sum_{l=0}^{L-1} |\Phi|_1 + \mu_2 \sum_{l=0}^{L-1} S(\Phi_l) \right)
#| $$
#| 
#| ... where $\mu_1$, $\mu_2$ are relative magnitudes usually set to 1, and
#| $\lambda$ controls overall regularization magnitude.
#| 
#| Their original implementation is 
#| 
#|  ```python3
#| def reg(acts_scale):
#| 
#|     def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
#|         return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)
#| 
#|     reg_ = 0.
#|     for i in range(len(acts_scale)):
#|         vec = acts_scale[i].reshape(-1, )
#| 
#|         p = vec / torch.sum(vec)
#|         l1 = torch.sum(nonlinear(vec))
#|         entropy = - torch.sum(p * torch.log2(p + 1e-4))
#|         reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy
#| 
#|     # regularize coefficient to encourage spline to be zero
#|     for i in range(len(self.act_fun)):
#|         coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
#|         coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
#|         reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1
#| 
#|     return reg_
#|  ```
#| 
#| However, this implementation makes use of their own way of simulating this.

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


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers : List[int],
        G: int = 5,
        K: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        b : torch.nn.Module =torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range=[-1, 1],
    ):

        super(KAN, self).__init__()
        self.G: int = G
        self.K: int = K

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers, layers[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    G=G,
                    K=K,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    b=b,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

#| 
#| Fairly simple application of KAT theorem laid out in the paper, i.e.
#| 
#| $$
#|     f(x) \approx \Phi_L \circ \Phi_{L-1} \circ \dots \circ \Phi_1(X)
#| $$
#| , with $X \in \mathbb{R}^{B \times I}$
#| 
#| 

def forward(self, x: torch.Tensor, update_grid=False):
    for layer in self.layers:
        if update_grid:
            layer.update_grid(x)
        x = layer(x)
    return x

#Dynamically assign the function to the class
KAN.forward = forward

# -------------------------------

#| Fairly simple function that sums up the regularization of all the layers
#| roughly following the formula seen in equation (2.20) of the paper, i.e.
#| $$
#| \ell_{\text{total}} = \ell_{\text{pred}} + \lambda \left( \mu_1 \sum_{l=0}^{L-1} |\Phi|_1 + \mu_2 \sum_{l=0}^{L-1} S(\Phi_l) \right)
#| $$

def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
    return sum(
        layer.regularization_loss(regularize_activation, regularize_entropy)
        for layer in self.layers
    )

#Dynamically assign the function to the class
KAN.regularization_loss = regularization_loss

# -------------------------------


#| ### Things to add to this notebook
#| 
#| 
#| #### Interpretability features
#| 
#| 
#| The original code for plotting the graphs showing how the splines replicate
#| univariate functions at different points is unfortunately not organized in the most cohesive manner.
#| In addition, this implementation focuses on efficiency, so it does not store the
#| preactivations or postactivations of the layers. This could be improved upon in
#| the future.
#| 
#| #### More stable curve fitting
#| 
#| Sometimes `torch.linalg.lstsq` can produce `NaN`s during its execution and
#| halt the training. I could look into why this happens with this code.

#| ## Toy Examples
#| 1. Test of evaluating $\text{exp} \left\{ \frac{x + y}{1 + x \cdot y} \right\}$ in region of $y \in [-1, 1]$ and $x \in [-1,1]$

def test_mul(kan : KAN):
    optimizer = torch.optim.LBFGS(kan.parameters(), lr=1)
    with tqdm(range(100)) as pbar:
        for i in pbar:
            loss, reg_loss = None, None
            if i > 4:
                break

            def closure():
                optimizer.zero_grad()
                x = torch.rand(1024, 2)
                y = kan(x, update_grid=(i % 20 == 0))

                assert y.shape == (1024, 1)
                nonlocal loss, reg_loss
                u = x[:, 0]
                v = x[:, 1]
                loss = nn.functional.mse_loss(y.squeeze(-1), (u + v) / (1 + u * v))
                reg_loss = kan.regularization_loss(1, 0)
                (loss + 1e-5 * reg_loss).backward()
                return loss + reg_loss

            optimizer.step(closure)
            pbar.set_postfix(mse_loss=loss.item(), reg_loss=reg_loss.item())
    for layer in kan.layers:
        print(layer.spline_weight)

# -------------------------------
try:
    kan = KAN([2, 2, 1], b=nn.Identity)
    test_mul(kan)
except Exception as e:
    print(e)

# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x : (x[:, 0] + x[:, 1]) /(1 + x[:, 0] *  x[:, 1])

train_x_ranges: Tuple = (-1,1)
train_y_ranges: Tuple = (-1,1)

n_intervals: int = 100
x_space: torch.tensor = torch.linspace(
    train_x_ranges[0], train_x_ranges[1], steps=n_intervals
)
y_space: torch.tensor = torch.linspace(
    train_y_ranges[0], train_y_ranges[1], steps=n_intervals
)

x_grid, y_grid = torch.meshgrid(x_space, y_space)
z_args: torch.tensor = torch.cat(
    (x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)), axis=1
)
x_grid = x_grid.reshape(-1, 1)
y_grid = y_grid.reshape(-1, 1)
z_evals: torch.tensor = f(z_args)
z_space: torch.tensor = z_evals.reshape(n_intervals, n_intervals)

train_xy: torch.tensor = torch.cat( (x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)), axis=1)

# -------------------------------
CommonUtils.plot_simple_3d_function_plotly(
    x_space,
    y_space,
    z_space,
    Path("./plots/exp_func_gt.html"),
    f"Plot of exponential function evaluation over range x:{train_x_ranges}, y:{train_y_ranges}", show=True
)
train_zout: torch.tensor = kan(train_xy)

# -------------------------------
CommonUtils.plot_simple_3d_function_plotly(
    x_space.detach(),
    y_space.detach(),
    train_zout.detach().reshape((n_intervals, n_intervals)),
    Path(
        f"./plots/model_eval.html"
    ),
    f"Plot of exponential function evaluation over range x:{train_x_ranges}, y:{train_y_ranges}", show=True
)
print("training stopped now")
