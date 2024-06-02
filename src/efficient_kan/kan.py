import torch
import torch.nn.functional as F
from typing import List
import math


class KANLinear(torch.nn.Module):
    """
    Some important definitions include

    $I \\in \\N$ , the input size of the layer, represented by `self.I` in the code

    $O \\in \\N$, the output size of the layer, represented by `self.O` in the code

    $G \\in \\N$, the number of grid intervals used in the spline, represented by `self.G` in the code

    $K \\in \\N$, the order of the spline to be used for activations, represented by `self.K` in the code

    The grid will be of shape  $I \\times (G + 2K + 1)$ and will be used to evaluate the spline, represented by `self.grid` in the code

    $b : \\R \\mapsto \\R$ will be a function applied elementwise to any matrix that serves as its input, represented by `self.b` in the code and having $\\text{SiLU}$ as its default value

    # TODO: explain the ifference between spline_weight and spline_scaler
    """

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
        self.b: torch.nn.Module = b()
        self.grid_eps: float = grid_eps

        self.init_parameters()

# grid is of shape I x (G + 2 * K + 1)
"""
#### Weight Initialization

Naturally, initializing the neurons to be the same value is redundant and
inefficient, since you'll have neurons that take on the same values and do the
same thing even after backpropagation. It's only natural that weights are
therefore intialized randomly. This has an effect on the convergence of the
network. However, some weight initialization schemes are more effective than others.

The rational behind the scaling of the layers is to account for the variance of the previous
layer and prevent the vanishing or explosion of the gradient in deeper networks.

Ideally, as reasoned in Gloriot & Bengio (2010), it is the case that

The intuition then goes that having the updates spread out in the same way would
mean that the layers are learning at the same rythm, thus mitigating the effects
of vanishing or exploding gradients.
(obtained from [here](https://datascience.stackexchange.com/questions/82917/why-do-we-want-the-variance-of-the-layers-to-remain-the-same-throughout-a-deep-n))

Thus, if that is the case, then:

$$
\text{Layer 1}\rightarrow \frac{\partial Cost}{\partial W^i}= \frac{\partial Cost}{\partial s^i}\frac{\partial s^i}{\partial W^i}=\frac{\partial Cost}{\partial s^i} (z^{i-1})^T \\
\text{Layer 2}\rightarrow \frac{\partial Cost}{\partial W^{i+1}}= \frac{\partial Cost}{\partial s^{i+1}}\frac{\partial s^{i+1}}{\partial W^{i+1}}=\frac{\partial Cost}{\partial s^{i+1}} (z^{i})^T
$$

Thus, considering that if:
$$
Var\left(\frac{\partial Cost}{\partial s^i}\right) = Var\left(\frac{\partial Cost}{\partial s^{i+1}}\right) \,\,\,\,\,\leftrightarrow\,\,\,\,\, Var(z^i)=Var(z^{i-1})
$$

then, this would mean:

$$
Var\left(\frac{\partial Cost}{\partial W^i} \right) = Var\left(\frac{\partial Cost}{\partial W^{i+1}} \right)
$$

Hence why initializing the weights to have the same variance is important.

#### Xavier Initialization

In the original KAN paper, they used Xavier initialization.
The initialization technique is listed as the second implementation detail of the
paper.

> Initialization scales. Each activation function is initialized to have
spline(x) ≈ 0 2. w is initialized according to the Xavier initialization, which
has been used to initialize linear layers in MLPs.


A very abbreviated rundown of the two goes as follows:

Xavier initialization, scales the weights proportional to the number of inputs to the layer.

This means that assuming a bias of zero, you initialize your weights to be 

$$
W_{i,j}^{(l)} \sim N(\mu = 0, \sigma^2 = \frac{1}{\sqrt{m^{(l-1)}}})
$$
... where $m$ is the number of input units to the next layer.

Note that sometimes $m^{l-1} + m^{l}$ is used.

However, one assumption of Xavier intialization was a derivative of 1 for the
activation function and mean of 0 for activations.

[Helpful video](https://www.youtube.com/watch?v=ScWTYHQra5E)

#### Kaiming He Intialization

The difference between Kaiming He initialization and Xavier intialization is
that Kaiming He takes into account the use of ReLU units. The approach is fairly
similar to Xavier intialization. The end result is a scaling factor that looks
like the following


$$
W_{i,j}^{(l)} \sim N(\mu = 0, \sigma^2 = \sqrt{\frac{2}{m^{(l-1)}}})
$$

"""
def init_parameters(self):
    """
    Uses Kaiming He initialization,
    a technique mainly used for initializing the weights of ReLU neural networks

    It initializes the weights such that the variance of the weights is the same across the network.

    The weights are initialized from a normal distribution $\mathcal{N}(0, \frac{2}{n}$

    It is related to the Xavier intialization, which is more suited for activation functions like sigmoid or tanh functions.
    """
    torch.nn.init.kaiming_uniform_(
        self.base_weight, a=math.sqrt(5) * self.scale_base
    )
    with torch.no_grad():
        noise = (
            (torch.rand(self.G + 1, self.I, self.O) - 1 / 2)
            * self.scale_noise
            / self.G
        )

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
    """
    Used for evaluating spline curves in B-spline form.
    """
    """
    Compute the B-spline bases for the given input tensor using the de Boor algorithm
    (Equivalent to coef2curve in the original repo)

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).

    Returns:
        torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
    """
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
    """
    The solution below computes a solution to the linear system

    $$
        \\underset{X \\in \\mathbb{K}}{||AX - B||_{F}}
    $$

    ... where in this case, $A$ is taken to be
    """
    """
    Compute the coefficients of the curve that interpolates the given points.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

    Returns:
        torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
    """
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
    """Returns the scaled spline weight. In some of the other repositories, this is optional

    Returns:
        torch.tensor: the self.spline_weight tensor but scaled
    """
    return self.spline_weight * (self.spline_scaler.unsqueeze(-1))


#Dynamically assign the function to the class
KANLinear.scaled_spline_weight = scaled_spline_weight


def forward(self, x: torch.Tensor):
    """
    In this function,
    $X \\in \\R^{B \\times I}$, where $B \\in \\N$ is the batch size, and $I \\in N$ is the input dimension that the layer takes in
    """
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

"""
This is noted in the paper (Liu et al.) as being one of the key implementation
details to making 
KANs optimizable. The grid update step is one of these implementation details.

> Update of spline grids. We update each grid on the fly according to its input
activations, to address the issue that splines are defined on bounded regions
but activation values can evolve out of the fixed region during training 3.

NOTE that this means that the splines will still evaluate, but it's likely that
the result will not be favorable due to the coefficients of the splines not
necessarily accounting for that input region.

"""
@torch.no_grad()
def update_grid(self, x: torch.Tensor, margin=0.01):
    """Update grid

    Args:
        x (torch.Tensor): input tensor to base the update off of
        margin (float, optional): Margin for the uniform extension of the grid. Defaults to 0.01.
    """
    assert x.dim() == 2 and x.size(1) == self.I
    batch = x.size(0)

    splines = self.b_splines(x)  # (batch, in, coeff)
    splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
    orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
    orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
    unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
    unreduced_spline_output = unreduced_spline_output.permute(
        1, 0, 2
    )  # (batch, in, out)

    # sort each channel individually to collect data distribution
    x_sorted = torch.sort(x, dim=0)[0]
    grid_adaptive = x_sorted[
        torch.linspace(0, batch - 1, self.G + 1, dtype=torch.int64, device=x.device)
    ]

    uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.G
    grid_uniform = (
        torch.arange(self.G + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
        * uniform_step
        + x_sorted[0]
        - margin
    )

    grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
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

    self.grid.copy_(grid.T)
    self.spline_weight.data.copy_(
        self.coefs_from_curve_data(x, unreduced_spline_output)
    )

#Dynamically assign the function to the class
KANLinear.update_grid = update_grid

def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
    """
    Compute the regularization loss.

    This is a dumb simulation of the original L1 regularization as stated in the
    paper, since the original one requires computing absolutes and entropy from the
    expanded (batch, in_features, out_features) intermediate tensor, which is hidden
    behind the F.linear function if we want an memory efficient implementation.

    The L1 regularization is now computed as mean absolute value of the spline
    weights. The authors implementation also includes this term in addition to the
    sample-based regularization.
    """
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

    """
        Fairly simple application of KAT theorem laid out in the paper, i.e.

        $$
            f(x) \\approx \\Phi_L \\circ \\Phi_{L-1} \\circ \\dots \\circ \\Phi_1(X)
        $$
        , with $X \\in \\R^{B \\times I}$

    """

def forward(self, x: torch.Tensor, update_grid=False):
    for layer in self.layers:
        if update_grid:
            layer.update_grid(x)
        x = layer(x)
    return x


#Dynamically assign the function to the class
KAN.forward = forward


def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
    return sum(
        layer.regularization_loss(regularize_activation, regularize_entropy)
        for layer in self.layers
    )

#Dynamically assign the function to the class
KAN.regularization_loss = regularization_loss