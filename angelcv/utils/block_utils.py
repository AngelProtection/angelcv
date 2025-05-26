import inspect

from einops import rearrange, repeat
import torch
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t


def get_block_name_to_impl_dict() -> dict[str, nn.Module]:
    """
    Creates a mapping between neural network block names and their implementations.

    Dynamically scans the blocks module and creates a dictionary that converts block
    names to their corresponding class implementations. Only includes classes that
    inherit from nn.Module.

    Returns:
        dict[str, nn.Module]: Dictionary mapping block names to their implementations

    Example:
        >>> block_map = get_block_name_to_impl_dict()
        >>> block_map
        {
            'ConvBlock': <class 'angelcv.model.blocks.ConvBlock'>,
            'ResidualBlock': <class 'angelcv.model.blocks.ResidualBlock'>,
            'CSPBlock': <class 'angelcv.model.blocks.CSPBlock'>,
            'TransformerBlock': <class 'angelcv.model.blocks.TransformerBlock'>
        }

        # Convert block name to its implementation
        >>> block_impl = block_map["ConvBlock"]
        >>> block = block_impl(in_channels=64, out_channels=128)
        >>> upsample_impl = block_map["nn.Upsample"]
        >>> upsample = upsample_impl(scale_factor=2, mode="nearest")
    """

    # NOTE: that if the blocks are implemented in multiple files this will need to be handled here!

    name_to_impl_map = {}
    # Start with native nn modules to the mapping
    import torch.nn as nn

    for name, block_impl in inspect.getmembers(nn, inspect.isclass):
        if issubclass(block_impl, nn.Module) and block_impl is not nn.Module:
            name_to_impl_map[f"nn.{name}"] = block_impl

    # Start with the blocks defined here
    from angelcv.model import blocks

    for name, block_impl in inspect.getmembers(blocks, inspect.isclass):
        if issubclass(block_impl, nn.Module) and block_impl is not nn.Module:
            name_to_impl_map[name] = block_impl

    return name_to_impl_map


# NOTE: The function accepts **kwargs for compatibility with Conv2dNormAct's usage pattern.
# When called with kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs)),
# any 'dilation' parameter in kwargs will be automatically passed to the dilation parameter.
# This allows auto_pad to calculate correct padding when dilation is specified in the convolution.
def auto_pad(kernel_size: _size_2_t, dilation: _size_2_t = 1, **kwargs) -> tuple[int, int]:
    """
    Auto Padding for the convolution blocks
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)


def create_activation_function(activation: str | None, inplace: bool = True) -> nn.Module:
    """
    Returns the activation function that matches it's input string (case-insensitive).
    """
    if not activation or activation.lower() in ["false", "none"]:
        return nn.Identity()

    activation_map = {
        name.lower(): obj
        for name, obj in nn.modules.activation.__dict__.items()
        if isinstance(obj, type) and issubclass(obj, nn.Module)
    }
    if activation.lower() in activation_map:
        return activation_map[activation.lower()](inplace=inplace)
    else:
        raise ValueError(f"Activation function '{activation}' is not found in torch.nn")


def generate_cell_centers_and_strides(
    feature_maps: list[Tensor],
    feature_map_strides: list[int],
) -> tuple[Tensor, Tensor]:
    """
    Generates a grid of centers for each feature map level and associates each cell center
    with its corresponding stride value.

    Args:
        feature_maps: List of feature tensors from different pyramid levels (shape: [B, C, H, W])
        feature_map_strides: Corresponding downsampling stride for each feature level
                        (e.g., [8, 16, 32] for 3 levels)

    Returns:
        tuple containing:
        - cell_centers: Tensor of shape (N, 2) with coordinates (x, y) for all cell centers
          For a pyramid with strides [8, 16, 32] and input size 640x640:
          - First level (stride=8): 80x80 grid:
            (0.5, 0.5), (1.5, 0.5), ..., (79.5, 0.5), (0.5, 1.5), ..., (79.5, 79.5) → 6400 centers
          - Second level (stride=16): 40x40 grid: similar pattern up to (39.5, 39.5) → 1600 centers
          - Third level (stride=32): 20x20 grid: similar pattern up to (19.5, 19.5) → 400 centers
        - cell_strides: Tensor of shape (N, 1) with stride value for each cell center
    """
    cell_center_offset_x = 0.5
    cell_center_offset_y = 0.5
    cell_centers = []
    cell_strides = []

    # Get dtype and device from first feature map
    dtype = feature_maps[0].dtype
    # NOTE: it's required to set the device manually as lightning can't hanlde the allocation
    # because it's generated dynamically
    device = feature_maps[0].device

    for level_idx, stride in enumerate(feature_map_strides):
        # Get height and width of current feature level
        _, _, grid_height, grid_width = feature_maps[level_idx].shape

        # Generate coordinate grids with offset
        x_coords = torch.arange(grid_width, device=device, dtype=dtype) + cell_center_offset_x
        y_coords = torch.arange(grid_height, device=device, dtype=dtype) + cell_center_offset_y

        # Create coordinate grid using einops
        x_grid = repeat(x_coords, "w -> h w", h=grid_height)
        y_grid = repeat(y_coords, "h -> h w", w=grid_width)

        # Stack and reshape coordinates
        cell_centers.append(rearrange(torch.stack((x_grid, y_grid)), "c h w -> (h w) c"))

        # Create stride tensor for this level
        cell_strides.append(torch.full((grid_height * grid_width, 1), stride, dtype=dtype, device=device))

    return torch.cat(cell_centers), torch.cat(cell_strides)


def distribution_to_box(distribution: Tensor, cell_centers: Tensor, dim: int = -1):
    """
    Convert distribution (left, top, right, bottom) to bounding box (xyxy).

    Args:
        distribution (Tensor): Distributions from cell centers to box edges.
        cell_centers (Tensor): Cell centers for the bounding boxes.
        dim (int): Dimension along which to chunk the distribution tensor.

    Returns:
        Tensor: Bounding boxes (xyxy)
    """
    # Split distribution into left-top and right-bottom components
    left_top, right_bottom = distribution.chunk(2, dim=dim)

    # Calculate top-left and bottom-right coordinates
    top_left = cell_centers - left_top
    bottom_right = cell_centers + right_bottom

    return torch.cat((top_left, bottom_right), dim=dim)


def box_to_distribution(bbox: Tensor, cell_centers: Tensor, max_distribution: float, dim: int = -1):
    """
    Convert bounding box (xyxy) to distribution (left, top, right, bottom).

    Args:
        cell_centers (Tensor): Cell centers for the bounding boxes.
        bbox (Tensor): Bounding boxes in xyxy format.
        max_distribution (float): Maximum distribution value for clamping.
        dim (int): Dimension along which to chunk the distribution tensor.

    Returns:
        Tensor: Distributions from cell centers to box edges.
    """
    # Split bbox into top-left and bottom-right coordinates
    top_left, bottom_right = bbox.chunk(2, dim=dim)

    # Calculate distributions from cell centers to box edges
    distributions = torch.cat((cell_centers - top_left, bottom_right - cell_centers), dim=dim)

    # Clamp distributions to be within the range [0, max_distribution)
    return distributions.clamp_(0, max_distribution - 1e-3)


if __name__ == "__main__":
    from pprint import pprint

    # START TEST: get_block_name_to_impl_dict
    block_map = get_block_name_to_impl_dict()
    pprint(block_map)
    # END TEST: get_block_name_to_impl_dict
