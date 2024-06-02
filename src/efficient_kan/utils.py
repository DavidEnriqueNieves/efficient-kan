from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Union, Optional, TypedDict, Any
from argparse import ArgumentParser, Namespace
import torch
from dataclasses import dataclass


@dataclass
class Dataset:
    train_input: torch.tensor
    train_label: torch.tensor
    test_input: torch.tensor
    test_label: torch.tensor

@dataclass
class Bounds:
    """
    Class for representing bounds of a function, mainly for notation and sanity checking
    """

    lower : float
    upper : float
    def __setattr__(self, __name: str, __value: Any) -> None:

        if isinstance(__value, int):
            __value = float(__value)

        assert isinstance(__value, float), f"{__name} must be an float"
            

        return super().__setattr__(__name, __value)
    
    def __post_init__(self) -> None:
        if isinstance(self.lower, int):
            self.lower = float(self.lower)
        if isinstance(self.upper, int):
            self.upper = float(self.upper)

        assert isinstance(self.lower, float), "lower bound must be an float"
        assert isinstance(self.upper, float), "upper bound must be an float"
        assert self.upper > self.lower, "upper bound must be greater than the lower bound!!!"

class CommonUtils:
    """
        Common Utilities class
    """
    @staticmethod
    def run_argparse() -> Namespace:

        argparser: ArgumentParser = ArgumentParser()
        argparser.add_argument("--debug", action="store_true")
        args: Namespace = argparser.parse_args()
        return args

    @staticmethod
    def run_debugger():
        import debugpy

        debug_port: int = 5678
        debugpy.listen(debug_port)
        print(f"Waiting for debugpy port at {debug_port}")
        debugpy.wait_for_client()

    # borrowed from them
    def create_dataset(
        f, n_var=2, ranges=[-1, 1], train_num=1000, test_num=1000, seed=0
    ) -> Dataset:

        torch.manual_seed(seed)

        train_input = torch.zeros(train_num, n_var)
        test_input = torch.zeros(test_num, n_var)
        for i in range(n_var):
            train_input[:, i] = (
                torch.rand(train_num) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]
            )
            test_input[:, i] = (
                torch.rand(test_num) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]
            )

        train_label = f(train_input)
        test_label = f(test_input)

        dataset: Dataset = Dataset(
            train_input=train_input,
            test_input=test_input,
            train_label=train_label,
            test_label=test_label,
        )

        return dataset

    def plot_losses(losses: List[float], path: Path, title: str):
        if not path.parent.exists():
            dir_str: str = str(path.parent.absolute())
            print(f"Making directory {dir_str}")
            os.mkdir(dir_str)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(losses)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(title)
        fig.savefig(path)

    @staticmethod
    def plot_simple_function(
        x: torch.tensor,
        y: torch.tensor,
        path: Path,
        title: str,
        axes_names: Tuple[str, str] = None,
        x_bounds: Tuple[float, float] = None,
        y_bounds: Tuple[float, float] = None,
    ):
        if not path.parent.exists():
            dir_str: str = str(path.parent.absolute())
            print(f"Making directory {dir_str}")
            os.mkdir(dir_str)
        min_x: float = torch.min(x)
        max_x: float = torch.max(x)

        min_y: float = torch.min(y)
        max_y: float = torch.max(y)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y)

        if x_bounds == None:
            x_bounds: Tuple = (min_x, max_x)
        if y_bounds == None:
            y_bounds: Tuple = (min_y, max_y)

        ax1.set_xlim(x_bounds[0], x_bounds[1])
        ax1.set_ylim(y_bounds[0], y_bounds[1])
        if axes_names != None:
            ax1.set_xlabel(axes_names[0])
            ax1.set_ylabel(axes_names[1])
        ax1.set_title(title)
        # fig.show()
        print("")
        fig.savefig(path)

    @staticmethod
    def plot_simple_3d_function_matplt(
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        path: Path,
        title: str,
        axes_names: Tuple[str, str, str] = None,
        x_bounds: Tuple[float, float] = None,
        y_bounds: Tuple[float, float] = None,
        z_bounds: Tuple[float, float] = None,
    ):
        if not path.parent.exists():
            dir_str: str = str(path.parent.absolute())
            print(f"Making directory {dir_str}")
            os.mkdir(dir_str)

        min_x: float = torch.min(x).item()
        max_x: float = torch.max(x).item()
        min_y: float = torch.min(y).item()
        max_y: float = torch.max(y).item()
        min_z: float = torch.min(z).item()
        max_z: float = torch.max(z).item()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the surface, mapping the z-values to a colormap
        surf = ax.plot_surface(x.numpy(), y.numpy(), z.numpy(), cmap="viridis")

        # Add a color bar to show the mapping
        fig.colorbar(surf, shrink=0.5, aspect=5)

        if x_bounds is None:
            x_bounds = (min_x, max_x)
        if y_bounds is None:
            y_bounds = (min_y, max_y)
        if z_bounds is None:
            z_bounds = (min_z, max_z)

        ax.set_xlim(x_bounds[0], x_bounds[1])
        ax.set_ylim(y_bounds[0], y_bounds[1])
        ax.set_zlim(z_bounds[0], z_bounds[1])

        if axes_names is not None:
            ax.set_xlabel(axes_names[0])
            ax.set_ylabel(axes_names[1])
            ax.set_zlabel(axes_names[2])

        ax.set_title(title)
        fig.savefig(path)

    @staticmethod
    def plot_simple_3d_function_plotly(
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        path: Path,
        title: str,
        axes_names: Tuple[str, str, str] = None,
        x_bounds: Bounds = None,
        y_bounds: Bounds = None,
        z_bounds: Bounds = None,
    ):
        if not path.parent.exists():
            dir_str: str = str(path.parent.absolute())
            print(f"Making directory {dir_str}")
            os.makedirs(dir_str, exist_ok=True)

        min_x: float = torch.min(x).item()
        max_x: float = torch.max(x).item()
        min_y: float = torch.min(y).item()
        max_y: float = torch.max(y).item()
        min_z: float = torch.min(z).item()
        max_z: float = torch.max(z).item()

        # Create the plotly figure
        fig = go.Figure(
            data=[
                go.Surface(z=z.numpy(), x=x.numpy(), y=y.numpy(), colorscale="Viridis")
            ]
        )

        # Update the layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(
                    range=[min_x, max_x] if x_bounds is None else x_bounds,
                    title=axes_names[0] if axes_names else "X-axis",
                ),
                yaxis=dict(
                    range=[min_y, max_y] if y_bounds is None else y_bounds,
                    title=axes_names[1] if axes_names else "Y-axis",
                ),
                zaxis=dict(
                    range=[min_z, max_z] if z_bounds is None else z_bounds,
                    title=axes_names[2] if axes_names else "Z-axis",
                ),
            ),
        )

        # Save the plot to a file
        fig.write_html(str(path))
        print(f"Saving figure to {path}")

    # COOL IDEA BUT TOO MUCH WORK
    @staticmethod
    def plot_side_by_side(
        train_truth: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        train_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        path: Path,
        title1: str,
        title2: str,
        val_truth: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        val_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
        axes_names: Tuple[str, str, str] = None,
        x_bounds: Tuple[float, float] = None,
        y_bounds: Tuple[float, float] = None,
        z_bounds: Tuple[float, float] = None,
    ):
        if not path.parent.exists():
            dir_str: str = str(path.parent.absolute())
            print(f"Making directory {dir_str}")
            os.makedirs(dir_str, exist_ok=True)

        # Combine training and validation data to find overall minimum and maximum values
        x1, y1, z1 = train_truth
        x2, y2, z2 = train_out

        if val_truth is not None and val_out is not None:
            x1 = torch.cat([x1, val_truth[0]])
            y1 = torch.cat([y1, val_truth[1]])
            z1 = torch.cat([z1, val_truth[2]])
            x2 = torch.cat([x2, val_out[0]])
            y2 = torch.cat([y2, val_out[1]])
            z2 = torch.cat([z2, val_out[2]])

        # Calculate minimum and maximum values for axes
        min_x = min(torch.min(x1).item(), torch.min(x2).item())
        max_x = max(torch.max(x1).item(), torch.max(x2).item())

        min_y = min(torch.min(y1).item(), torch.min(y2).item())
        max_y = max(torch.max(y1).item(), torch.max(y2).item())

        min_z = min(torch.min(z1).item(), torch.min(z2).item())
        max_z = max(torch.max(z1).item(), torch.max(z2).item())

        # Update x_bounds, y_bounds, and z_bounds if not defined
        x_bounds = x_bounds if x_bounds is not None else (min_x, max_x)
        y_bounds = y_bounds if y_bounds is not None else (min_y, max_y)
        z_bounds = z_bounds if z_bounds is not None else (min_z, max_z)

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=(title1, title2)
        )

        # First surface plot
        fig.add_trace(
            go.Surface(x=train_truth[0].detach().numpy(), y=train_truth[1].detach().numpy(), z=train_truth[2].detach().numpy(), colorscale="Viridis", name=title1),
            row=1, col=1
        )

        # Second surface plot
        fig.add_trace(
            go.Surface(x=train_out[0].detach().numpy(), y=train_out[1].detach().numpy(), z=train_out[2].detach().numpy(), colorscale="Viridis", name=title2),
            row=1, col=2
        )

        # Update layout for side-by-side plots
        fig.update_layout(
            title_text=f"",
            scene=dict(
                xaxis=dict(
                    range=x_bounds,
                    title=axes_names[0] if axes_names else "X-axis",
                ),
                yaxis=dict(
                    range=y_bounds,
                    title=axes_names[1] if axes_names else "Y-axis",
                ),
                zaxis=dict(
                    range=z_bounds,
                    title=axes_names[2] if axes_names else "Z-axis",
                ),
            ),
            scene2=dict(
                xaxis=dict(
                    range=x_bounds,
                    title=axes_names[0] if axes_names else "X-axis",
                ),
                yaxis=dict(
                    range=y_bounds,
                    title=axes_names[1] if axes_names else "Y-axis",
                ),
                zaxis=dict(
                    range=z_bounds,
                    title=axes_names[2] if axes_names else "Z-axis",
                ),
            ),
        )

        # Save the plot to a file
        fig.write_html(str(path))
        print(f"Saving figure to {path}")
