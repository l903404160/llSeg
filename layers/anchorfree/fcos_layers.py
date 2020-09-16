import torch


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w*stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h*stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)
    shifts_x = shifts_x.reshape(-1)
    shifts_y = shifts_y.reshape(-1)
    locations = torch.stack((shifts_x, shifts_y), dim=1) + stride // 2
    return locations