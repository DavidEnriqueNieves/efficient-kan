from icecream import ic
import sys
# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append("../src/efficient_kan")
from kan import KANLinear, KAN
from argparse import ArgumentParser


"""
REALLY simple script just to get a KAN running and be able to run through everything

"""

if __name__ == "__main__":
    
    argparser : ArgumentParser = ArgumentParser()
    argparser.add_argument("--debug", action="store_true")
    args : dict = argparser.parse_args()
    if(args.debug):
        PORT: int = 5678
        import debugpy
        print(f"Waiting on port {PORT}")
        debugpy.listen(PORT)
        debugpy.wait_for_client()

    print("imported successfully")

    kan : KAN = KAN([3, 2, 1],
        G=5,
        K=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        b=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1])

    input : torch.tensor = torch.randn((7,3))

    print("Output is ")
    out : torch.tensor = kan.forward(input)
    ic(out)