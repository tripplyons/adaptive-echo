import torch


# t is a torch tensor in these function, a and b are floats, returns a torch tensor
def linear_interp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return a + (b - a) * t


def exp_interp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return a * (b / a) ** t
