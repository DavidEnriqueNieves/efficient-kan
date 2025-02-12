import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
from utils import CommonUtils
from pathlib import Path
from efficient_kan import KAN


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

if __name__ == "__main__":
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

    CommonUtils.plot_simple_3d_function_plotly(
        x_space,
        y_space,
        z_space,
        Path("./plots/exp_func_gt.html"),
        f"Plot of exponential function evaluation over range x:{train_x_ranges}, y:{train_y_ranges}",
    )
    train_zout: torch.tensor = kan(train_xy)

    CommonUtils.plot_simple_3d_function_plotly(
        x_space.detach(),
        y_space.detach(),
        train_zout.detach().reshape((n_intervals, n_intervals)),
        Path(
            f"./plots/model_eval.html"
        ),
        f"Plot of exponential function evaluation over range x:{train_x_ranges}, y:{train_y_ranges}",
    )
    print("training stopped now")