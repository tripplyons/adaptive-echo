import torch


# t is a torch tensor in these function, a and b are floats, returns a torch tensor
def linear_interp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return a + (b - a) * t


def exp_interp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    EPSILON = 1e-6

    a_clamped = torch.clamp(torch.abs(a), EPSILON, None) * torch.sign(a)
    b_clamped = torch.clamp(torch.abs(b), EPSILON, None) * torch.sign(b)

    ratio = b_clamped / a_clamped
    ratio = torch.clamp(ratio, EPSILON, 1.0 / EPSILON)

    return a_clamped * torch.pow(ratio, t)
