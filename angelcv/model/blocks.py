import copy
from typing import Literal

from einops import rearrange
import torch
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t

from angelcv.utils.block_utils import (
    auto_pad,
    create_activation_function,
    distribution_to_box,
    generate_cell_centers_and_strides,
)


# ------------------------------- GENERAL -------------------------------
class Conv2dNormAct(nn.Module):
    """Basic Conv2d -> BatchNorm2d -> Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        *,
        activation: str | None = "SiLU",
        **kwargs,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            activation: Name of the activation function. Defaults to "SiLU".
            **kwargs: Additional arguments passed to `nn.Conv2d`.
        """
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            create_activation_function(activation),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Conv2d -> Norm -> Activation."""
        return self.layers(x)


class Concat(nn.Module):
    """Concatenates a list of tensors along a specified dimension."""

    def __init__(self, dim: int = 1):
        """
        Args:
            dim: Dimension along which to concatenate. Defaults to 1.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: tuple[Tensor, ...] | list[Tensor]) -> Tensor:
        """Concatenates the input tensors."""
        return torch.cat(x, dim=self.dim)


class Bottleneck(nn.Module):
    """
    Standard bottleneck block with optional residual connection.

    Reduces channels -> Processes -> Expands channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,  # Forces following arguments to be keyword-only
        reduction_kernel_size: int = 3,
        expansion_kernel_size: int = 3,
        use_residual: bool = True,
        expansion_ratio: float = 0.5,
        groups: int = 1,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            reduction_kernel_size: Kernel size for the reduction convolution. Defaults to 3.
            expansion_kernel_size: Kernel size for the expansion convolution. Defaults to 3.
            use_residual: If True and `in_channels == out_channels`, adds a residual connection. Defaults to True.
            expansion_ratio: Ratio to determine the intermediate bottleneck channels. Defaults to 0.5.
            groups: Number of groups for the expansion convolution. Defaults to 1.
        """
        super().__init__()

        # Calculate the number of channels in the reduced representation
        neck_channels = int(out_channels * expansion_ratio)

        # Validate parameters
        if groups > 1:
            assert neck_channels % groups == 0, (
                f"Neck channels ({neck_channels}) must be divisible by groups ({groups})"
            )
            assert out_channels % groups == 0, (
                f"Output channels ({out_channels}) must be divisible by groups ({groups})"
            )

        self.bottleneck = nn.Sequential(
            # First convolution: reduce dimensionality
            Conv2dNormAct(in_channels, neck_channels, kernel_size=reduction_kernel_size, activation="SiLU"),
            # Second convolution: process and expand back to original dimensionality
            Conv2dNormAct(
                neck_channels, out_channels, kernel_size=expansion_kernel_size, groups=groups, activation="SiLU"
            ),
        )

        # Enable residual connection only if dimensions match and it's requested
        self.use_residual = use_residual
        if self.use_residual:
            assert in_channels == out_channels, "Residual connection requires in_channels == out_channels"

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the bottleneck block."""
        out = self.bottleneck(x)

        if self.use_residual:
            out = out + x

        return out


class Attention(nn.Module):
    """
    Self-Attention module that enhances feature representations through multi-head attention.

    This module implements a self-attention mechanism that allows the model to weigh the importance
    of different spatial locations in the input. It processes the input through:
    1. Computing Query, Key, and Value representations using convolutions
    2. Applying multi-head attention mechanism
    3. Adding positional encoding
    4. Final projection to output dimensions

    The module combines both channel and spatial attention aspects, making it effective for
    capturing both local and global dependencies in the feature maps.
    """

    def __init__(self, in_channels: int, *, num_heads: int = 8, attention_ratio: float = 0.5):
        """
        Args:
            in_channels: Number of input channels.
            num_heads: Number of attention heads. Defaults to 8.
            attention_ratio: Ratio determining the key dimension relative to head dimension. Defaults to 0.5.
        """
        super().__init__()

        # Calculate dimensions for attention mechanism
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads  # Dimension per attention head
        self.key_dim = int(self.head_dim * attention_ratio)  # Reduced key dimension
        self.attention_scale = self.key_dim**-0.5  # Scaling factor for dot products

        # Calculate total channels needed for QKV computation
        total_key_dim = self.key_dim * num_heads  # Total dimension across all heads
        qkv_total_channels = in_channels + (total_key_dim * 2)  # Channels for query, key, and value

        # Define the main attention components
        self.qkv_projection = Conv2dNormAct(in_channels, qkv_total_channels, kernel_size=1, activation=None)

        # Depthwise convolution for position-sensitive encoding
        self.positional_encoder = Conv2dNormAct(
            in_channels, in_channels, kernel_size=3, groups=in_channels, activation=None
        )

        self.output_projection = Conv2dNormAct(in_channels, in_channels, kernel_size=1, activation=None)

    def forward(self, x: Tensor) -> Tensor:
        """Applies multi-head self-attention to the input tensor."""
        batch_size, channels, height, width = x.shape

        # Project input to query, key, and value representations
        qkv_combined = self.qkv_projection(x)

        # Clearer reshaping using einops
        qkv_reshaped = rearrange(
            qkv_combined,
            "b (num_heads d) h w -> b num_heads d (h w)",
            num_heads=self.num_heads,
            d=(self.key_dim * 2 + self.head_dim),  # Total dim per head
        )

        # Split into separate query, key, and value tensors
        query, key, value = qkv_reshaped.split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        # Compute scaled dot-product attention
        # Shape: [batch, heads, key_dim, height*width]
        attention_scores = (rearrange(query, "b h d hw -> b h hw d") @ key) * self.attention_scale
        attention_weights = attention_scores.softmax(dim=-1)

        # Apply attention weights to values
        # Shape: [batch, heads, head_dim, height*width]
        attended_values = value @ rearrange(attention_weights, "b h hw1 hw2 -> b h hw2 hw1")

        # Reshape back to spatial dimensions
        attended_features = rearrange(attended_values, "b num_heads d (h w) -> b (num_heads d) h w", h=height, w=width)

        # Add positional encoding
        value_spatial = rearrange(value, "b num_heads d (h w) -> b (num_heads d) h w", h=height, w=width)
        positional_features = self.positional_encoder(value_spatial)
        enhanced_features = attended_features + positional_features

        # Final projection
        return self.output_projection(enhanced_features)


# ------------------------------- SHARED --------------------------------


class C2f(nn.Module):
    """
    CSP Bottleneck (Cross Stage Partial) Fast, with 2 convolutions.

    This module enhances the CSP (Cross Stage Partial) Bottleneck architecture by using two
    convolutions and multiple bottleneck blocks. It splits the input channels, processes them
    through parallel paths, and merges them back using concatenation and a final convolution.

    The module consists of three main components:
    1. Initial split convolution that divides features into two paths
    2. Multiple bottleneck blocks processing one of the paths
    3. Final merge convolution that combines all paths
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,  # Forces following arguments to be keyword-only
        repeats: int = 1,
        use_residual: bool = False,
        groups: int = 1,
        expansion_ratio: float = 0.5,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            repeats: Number of bottleneck blocks. Defaults to 1.
            use_residual: Whether bottlenecks use residual connections. Defaults to False.
            groups: Number of groups for bottleneck convolutions. Defaults to 1.
            expansion_ratio: Channel expansion ratio for the hidden layers. Defaults to 0.5.
        """
        super().__init__()

        # Calculate hidden channels based on reduction ratio
        self.neck_channels = int(out_channels * expansion_ratio)

        # Initial convolution that splits the input into two paths
        self.split_conv = Conv2dNormAct(in_channels, 2 * self.neck_channels, kernel_size=1)

        # Create bottleneck blocks
        self.bottlenecks = nn.ModuleList(
            Bottleneck(
                self.neck_channels,
                self.neck_channels,
                use_residual=use_residual,
                groups=groups,
                expansion_ratio=1.0,  # No expansion within bottlenecks
                reduction_kernel_size=3,
                expansion_kernel_size=3,
            )
            for _ in range(repeats)
        )

        # Final convolution that merges all paths
        # Input channels = 2 paths + outputs from num_bottlenecks blocks
        self.merge_conv = Conv2dNormAct((2 + repeats) * self.neck_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the C2f module."""
        # Split features into two paths
        split_features = list(self.split_conv(x).chunk(2, dim=1))

        # Process one path through bottleneck blocks
        for bottleneck in self.bottlenecks:
            split_features.append(bottleneck(split_features[-1]))

        # Merge all paths
        return self.merge_conv(torch.cat(split_features, dim=1))


class SCDown(nn.Module):
    """
    Separable Convolution Downsampling. Uses 1x1 conv + depthwise 3x3 conv with stride.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,  # Forces following arguments to be keyword-only
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 2,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for the depthwise downsampling convolution. Defaults to 3.
            stride: Stride for downsampling. Defaults to 2.
        """
        super().__init__()

        self.layers = nn.Sequential(
            # 1x1 convolution for channel transformation (reduce number of channels)
            Conv2dNormAct(in_channels, out_channels, kernel_size=1, activation="SiLU"),
            # Depthwise convolution for spatial downsampling
            Conv2dNormAct(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=out_channels,  # Makes it depthwise
                activation=None,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the SCDown module."""
        return self.layers(x)


# ------------------------------ BACKBONE -------------------------------
# NOTE: no backbone specific blocks

# -------------------------------- NECK ---------------------------------


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling Fast (SPPF) for multi-scale feature aggregation.

    Processes input through multiple max-pooling layers and concatenates results
    to capture features at different scales.

    Structure:
    1. Channel reduction -> 2. Three sequential pooling steps -> 3. Concatenation -> 4. Final projection
    """

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 5, reduction_ratio: float = 0.5):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for the MaxPool layers. Defaults to 5.
            reduction_ratio: Ratio to reduce channels before pooling. Defaults to 0.5.
        """
        super().__init__()

        # Calculate hidden channels for dimensionality reduction
        hidden_channels = int(in_channels * reduction_ratio)

        # Initial 1x1 channel reduction
        self.reduction_conv = Conv2dNormAct(in_channels, hidden_channels, kernel_size=1, activation="SiLU")

        # Shared max-pool layer (same parameters for all pooling steps)
        # padding = k//2 maintains spatial dimensions
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        # Final 1x1 convolution to process concatenated features
        # Input channels = hidden_channels * 4 (original + 3 pooled versions)
        self.projection_conv = Conv2dNormAct(hidden_channels * 4, out_channels, kernel_size=1, activation="SiLU")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the SPPF module."""
        # Feature reduction
        branch1 = self.reduction_conv(x)

        # Create other branches with progressive pooling
        branch2 = self.max_pool(branch1)
        branch3 = self.max_pool(branch2)
        branch4 = self.max_pool(branch3)

        # Concatenate all branches and project channels
        return self.projection_conv(torch.cat([branch1, branch2, branch3, branch4], dim=1))


class PSA(nn.Module):
    """
    Position-Sensitive Attention (PSA) processes features through:
    1. Split -> 2. Parallel Path Processing -> 3. Merge

    - Split: Divides features into two equal paths
    - Process: Enhances one path with attention + feed-forward network
    - Merge: Combines paths and projects to original shape

    Maintains identical input/output channels for residual connections.
    """

    def __init__(self, in_channels: int, out_channels: int, *, expansion_ratio: float = 0.5, min_heads: int = 64):
        """
        Args:
            in_channels: Input feature channels (must match out_channels).
            out_channels: Output feature channels (must match in_channels).
            expansion_ratio: Hidden layer size ratio for splitting. Defaults to 0.5.
            min_heads: Minimum channels per attention head. Defaults to 64.
        """
        super().__init__()
        assert in_channels == out_channels, "Input and output channels must be identical for PSA"

        # Channel reduction calculations
        hidden_channels = int(out_channels * expansion_ratio)

        # Feature splitting
        self.split_conv = Conv2dNormAct(
            in_channels,
            2 * hidden_channels,  # Split into two paths
            kernel_size=1,
            activation="SiLU",
        )

        # Attention module for position-sensitive processing
        self.attention = Attention(hidden_channels, num_heads=max(1, hidden_channels // min_heads), attention_ratio=0.5)

        # Feed-forward network for feature enhancement (expand -> contract)
        self.feed_forward = nn.Sequential(
            Conv2dNormAct(
                hidden_channels,
                2 * hidden_channels,
                kernel_size=1,
                activation="SiLU",
            ),
            Conv2dNormAct(2 * hidden_channels, hidden_channels, kernel_size=1, activation=None),
        )

        # Feature merging (skip + processed features)
        self.output_projection = Conv2dNormAct(2 * hidden_channels, out_channels, kernel_size=1, activation="SiLU")

    def forward(self, x: Tensor) -> Tensor:
        """Process input through the split-process-merge pipeline."""
        # Split into two processing paths
        features = self.split_conv(x)
        skip_path, process_path = features.chunk(2, dim=1)

        # Enhance processing path with residuals
        process_path = process_path + self.attention(process_path)  # Attention residual
        process_path = process_path + self.feed_forward(process_path)  # FFN residual

        # Combine paths and project features
        return self.output_projection(torch.cat([skip_path, process_path], dim=1))


# ------------------------------- HEADS ---------------------------------


class RepVGGDW(nn.Module):
    """
    RepVGG Depthwise (RepVGGDW) module that implements efficient depthwise separable convolutions.

    This module implements a variant of the RepVGG architecture focusing on depthwise convolutions.
    It uses two parallel convolution paths:
    1. A 7x7 depthwise convolution for larger receptive field
    2. A 3x3 depthwise convolution for local feature extraction

    The outputs are combined additively and passed through activation.

    The module is particularly effective for spatial feature extraction while maintaining
    computational efficiency through depthwise convolutions and structural re-parameterization.
    """

    def __init__(self, channels: int, *, large_kernel_size: int = 7, small_kernel_size: int = 3):
        """
        Args:
            channels: Number of input/output channels (must be the same).
            large_kernel_size: Kernel size for the larger depthwise convolution. Defaults to 7.
            small_kernel_size: Kernel size for the smaller depthwise convolution. Defaults to 3.
        """
        super().__init__()

        # Small kernel depthwise convolution
        self.small_conv = Conv2dNormAct(
            channels,
            channels,
            kernel_size=small_kernel_size,
            stride=1,
            padding=small_kernel_size // 2,
            groups=channels,  # Makes it depthwise
            activation=None,  # No activation as it's applied after addition
        )

        # Large kernel depthwise convolution
        self.large_conv = Conv2dNormAct(
            channels,
            channels,
            kernel_size=large_kernel_size,
            stride=1,
            padding=large_kernel_size // 2,
            groups=channels,  # Makes it depthwise
            activation=None,  # No activation as it's applied after addition
        )

        # Activation function applied after addition
        self.activation = create_activation_function("SiLU")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass applying parallel depthwise convolutions."""
        return self.activation(self.large_conv(x) + self.small_conv(x))


class CIB(nn.Module):
    """
    Channel Information Block (CIB) that enhances feature processing through grouped convolutions.

    This module implements an advanced feature processing block that combines:
    1. Depthwise spatial processing
    2. Channel mixing and expansion
    3. Optional local attention through RepVGGDW
    4. Channel projection and refinement

    The module is particularly effective at capturing both spatial and channel relationships
    while maintaining computational efficiency through grouped convolutions and optional
    residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,  # Forces following arguments to be keyword-only
        use_residual: bool = True,
        expansion_ratio: float = 0.5,
        use_local_key: bool = False,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            use_residual: If True and `in_channels == out_channels`, adds a residual connection. Defaults to True.
            expansion_ratio: Ratio for channel expansion in hidden layers. Defaults to 0.5.
            use_local_key: If True, uses `RepVGGDW` for local processing. Defaults to False.
        """
        super().__init__()

        # Enable residual connection only if dimensions match and it's requested
        self.use_residual = use_residual and in_channels == out_channels

        # Calculate expanded channels for hidden layers
        expanded_channels = int(out_channels * expansion_ratio)
        hidden_channels = 2 * expanded_channels

        # Main processing sequence
        self.processing_layers = nn.Sequential(
            # Spatial processing with depthwise convolution
            Conv2dNormAct(in_channels, in_channels, kernel_size=3, groups=in_channels, activation="SiLU"),
            # Channel mixing and expansion
            Conv2dNormAct(in_channels, hidden_channels, kernel_size=1, activation="SiLU"),
            # Local key processing or standard depthwise convolution
            RepVGGDW(hidden_channels)
            if use_local_key
            else Conv2dNormAct(
                hidden_channels, hidden_channels, kernel_size=3, groups=hidden_channels, activation="SiLU"
            ),
            # Channel projection
            Conv2dNormAct(hidden_channels, out_channels, kernel_size=1, activation="SiLU"),
            # Final spatial refinement
            Conv2dNormAct(out_channels, out_channels, kernel_size=3, groups=out_channels, activation="SiLU"),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the CIB module."""
        output = self.processing_layers(x)
        if self.use_residual:
            output += x
        return output


class C2fCIB(C2f):
    """
    Cross-Concat Feed Forward with Channel Information Block (C2fCIB) that enhances feature processing.

    This module extends the C2f architecture by replacing standard bottlenecks with Channel
    Information Blocks (CIB). It maintains the split-transform-merge strategy while adding:
    1. Enhanced channel interaction through CIB modules
    2. Optional local key attention mechanisms
    3. Flexible channel scaling through expansion ratios

    The module is particularly effective for tasks requiring strong channel-wise feature
    relationships while maintaining the computational efficiency of the C2f architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,  # Forces following arguments to be keyword-only
        repeats: int = 1,
        use_residual: bool = False,
        use_local_key: bool = False,
        groups: int = 1,
        expansion_ratio: float = 0.5,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            repeats: Number of CIB blocks. Defaults to 1.
            use_residual: Whether CIB blocks use residual connections. Defaults to False.
            use_local_key: Whether CIB blocks use local key attention. Defaults to False.
            groups: Number of groups for convolutions (passed to parent C2f). Defaults to 1.
            expansion_ratio: Channel expansion ratio for hidden layers (passed to parent C2f). Defaults to 0.5.
        """
        # Initialize parent C2f module
        super().__init__(
            in_channels,
            out_channels,
            repeats=repeats,
            use_residual=use_residual,
            groups=groups,
            expansion_ratio=expansion_ratio,
        )

        # Replace bottlenecks with CIB modules
        self.bottlenecks = nn.ModuleList(
            CIB(
                self.neck_channels,
                self.neck_channels,
                use_residual=use_residual,
                expansion_ratio=1.0,  # Full channel expansion within CIB
                use_local_key=use_local_key,
            )
            for _ in range(repeats)
        )


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module that implements integral-based regression.

    This module implements the integral component of Distribution Focal Loss as proposed in
    'Generalized Focal Loss' (https://ieeexplore.ieee.org/document/9792391). It works by:
    1. Creating a fixed weight convolutional layer that acts as an integral operator
    2. Converting discrete probability distributions to continuous predictions
    3. Applying softmax normalization for probability distribution
    4. Computing the expected value through convolution

    The module is particularly effective for bounding box regression by treating it as
    a general distribution learning problem.
    """

    def __init__(self, num_channels: int = 16):
        """
        Args:
            num_channels: Number of discretization bins for regression output. Defaults to 16.
        """
        super().__init__()

        # Create fixed weight convolutional layer for integral operation
        self.integral_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False).requires_grad_(False)

        # Initialize weights as range values [0, 1, 2, ..., num_channels-1]
        weight_range = torch.arange(num_channels, dtype=torch.float)
        conv_weights = rearrange(weight_range, "wr -> 1 wr 1 1")
        self.integral_conv.weight.data[:] = nn.Parameter(conv_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes expected values from input distributions.

        Args:
            x: Input tensor (B, 4 * num_channels, num_cells) with probability distributions.

        Returns:
            Tensor (B, 4, num_cells) with expected coordinate values.
        """
        # Rearrange input from (B, 4C, A) -> (B, C, 4, A)
        distributions = rearrange(x, "b (n c) a -> b c n a", n=4)

        # Apply softmax along channel dimension to get probabilities
        probabilities = distributions.softmax(dim=1)

        # Apply integral convolution and reshape result
        # Conv2d expects 4D input (B, C, H, W)
        expected_values = self.integral_conv(probabilities)

        # 4. Rearrange output from (B, 1, 4, A) -> (B, 4, A)
        return rearrange(expected_values, "b 1 n a -> b n a")


class v10Detect(nn.Module):  # noqa: N801
    """
    YOLOv10 Detection Head for object detection tasks.

    This module implements the detection head from YOLOv10 (https://arxiv.org/pdf/2405.14458)
    with the following key components:
    1. Multiple detection branches for different scales
    2. Dual-path architecture with one-to-many and one-to-one predictions
    3. Efficient channel scaling based on class count
    4. Lightweight classification heads with grouped convolutions

    Example dimensions for 640x640 input image with 80 classes:
    - Input feature maps (P3, P4, P5):
      [(B, 256, 80, 80), (B, 512, 40, 40), (B, 1024, 20, 20)]
    - Output during training:
      {
        "one_to_many": [(B, 144, 80, 80), (B, 144, 40, 40), (B, 144, 20, 20)],
        "one_to_one": [(B, 144, 80, 80), (B, 144, 40, 40), (B, 144, 20, 20)]
      }
      where 144 = 80 classes + 16 channels * 4 coordinates
    - Output during inference:
      Tensor of shape (B, 300, 6) containing top 300 detections
      where 6 = (x1, y1, x2, y2, confidence, class_id)
    """

    def __init__(
        self, num_classes: int, num_dfl_bins: int, feature_map_strides: list[int], in_channels_list: list[int]
    ):
        """
        Args:
            num_classes: Number of object classes.
            num_dfl_bins: Number of distribution focal loss bins for regression output.
            feature_map_strides: List of feature map strides.
            in_channels_list: List of input channels for each detection layer (e.g., P3, P4, P5).
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_dfl_bins = num_dfl_bins
        self.feature_map_strides = torch.tensor(feature_map_strides)
        self.in_channels_list = in_channels_list

        self.current_stage: Literal["train", "validate", "test", "inference"] = "inference"

        # Detection settings
        self.detection_limit = 400
        self.previous_shape = None
        self.cell_centers = torch.empty(0)
        self.cell_strides = torch.empty(0)

        # DFL decoder
        self.dfl = DFL(self.num_dfl_bins)

        self._initialize_reg_heads()
        self._initialize_cls_heads()

    def _initialize_reg_heads(self):
        """Initialize the regression heads, using grouped convolutions."""
        # Determines intermediate channels for regression convs. It takes the greater of two values:
        # 1. `self.num_dfl_bins * 4`: Four times the DFL bins (e.g., 16 bins * 4 coords = 64),
        #    ensuring minimum capacity for DFL's output structure (predicting a distribution
        #    for each of the 4 bounding box coordinates).
        # 2. `self.in_channels_list[0] // 4`: A quarter of the channels from the first (highest
        #    resolution) input feature map. This scales capacity with the detail level of input
        #    features. The division by 4 acts as a common bottleneck, reducing computational
        #    load and encouraging a compact feature representation for regression (only used in yolov10x).
        # This approach adapts head capacity to both input feature richness and DFL complexity.
        reg_channels = max(self.num_dfl_bins * 4, self.in_channels_list[0] // 4)
        # n - 16 * 4 = 64 |  64 // 4 = 16 --> 64
        # s - 16 * 4 = 64 | 128 // 4 = 32 --> 64
        # m - 16 * 4 = 64 | 192 // 4 = 48 --> 64
        # b - 16 * 4 = 64 | 256 // 4 = 64 --> 64
        # l - 16 * 4 = 64 | 256 // 4 = 64 --> 64
        # x - 16 * 4 = 64 | 320 // 4 = 80 --> 80 (only one that uses self.in_channels_list[0] // 4)
        self.reg_head = nn.ModuleList(
            nn.Sequential(
                Conv2dNormAct(in_channels, reg_channels, kernel_size=3),
                Conv2dNormAct(reg_channels, reg_channels, kernel_size=3),
                nn.Conv2d(reg_channels, 4 * self.num_dfl_bins, kernel_size=1),
            )
            for in_channels in self.in_channels_list
        )

        # Dual path component
        self.one_to_one_reg_head = copy.deepcopy(self.reg_head)

    def _initialize_cls_heads(self):
        """Initialize the classification heads, using grouped convolutions."""
        # Determines intermediate channels for classification convs. It takes the greater of two values:
        # 1. `self.num_classes`: The number of object classes. This ensures the head has a
        #    minimum width appropriate for the number of categories to predict, especially
        #    if input channels are fewer than the class count (e.g., yolov10n).
        # 2. `self.in_channels_list[0]`: The channel count from the first (highest resolution)
        #    input feature map. This allows the head's capacity to scale with the richness
        #    of input features for larger models.
        # This approach ensures the head is sufficiently wide for the classification task
        # while also adapting to the detail level of input features.
        cls_channels = max(self.num_classes, self.in_channels_list[0])
        # (Assuming num_classes = 80)
        # n - max(80,  64) =  80
        # s - max(80, 128) = 128
        # m - max(80, 192) = 192
        # b - max(80, 256) = 256
        # l - max(80, 256) = 256
        # x - max(80, 320) = 320
        self.cls_head = nn.ModuleList(
            nn.Sequential(
                Conv2dNormAct(in_channels, in_channels, kernel_size=3, groups=in_channels),
                Conv2dNormAct(in_channels, cls_channels, kernel_size=1),
                Conv2dNormAct(cls_channels, cls_channels, kernel_size=3, groups=cls_channels),
                Conv2dNormAct(cls_channels, cls_channels, kernel_size=1),
                nn.Conv2d(cls_channels, self.num_classes, kernel_size=1),
            )
            for in_channels in self.in_channels_list
        )

        # Dual path component
        self.one_to_one_cls_head = copy.deepcopy(self.cls_head)

    def forward(self, x: list[Tensor]) -> dict | Tensor:
        """
        Forward pass of the detection head.

        Args:
            x (list[Tensor]): List of 3 feature maps from backbone/neck
                Example for 640x640 input (nano model):
                [
                    Tensor(B, 64, 80, 80),  # P3
                    Tensor(B, 128, 40, 40),  # P4
                    Tensor(B, 256, 20, 20)  # P5
                ]

        Returns:
            During training or when return_both_paths=True:
                dict: {
                    "one_to_many": List of 3 tensors each shaped (B, 144, H, W),
                    "one_to_one": List of 3 tensors each shaped (B, 144, H, W)
                }
                where 144 = num_classes + 16 channels * 4 coordinates

            During inference:
                Tensor: Shape (B, 300, 6) containing filtered predictions
                       where 6 = (x1, y1, x2, y2, confidence, class_id)
        """
        # Process detached features for one-to-one path
        one_to_one_output = [
            torch.cat(
                (
                    self.one_to_one_reg_head[idx](feature_map.detach()),
                    self.one_to_one_cls_head[idx](feature_map.detach()),
                ),
                dim=1,
            )
            for idx, feature_map in enumerate(x)
        ]

        # Process one-to-many path for train/validate/test stages
        if self.current_stage in ("train", "validate", "test"):
            # Process one-to-many path
            for i in range(len(self.in_channels_list)):
                x[i] = torch.cat((self.reg_head[i](x[i]), self.cls_head[i](x[i])), 1)

            # Return dictionary for train stages
            if self.current_stage == "train":
                return {"one_to_many": x, "one_to_one": one_to_one_output}

        processed_detections = self._process_filter_predictions(one_to_one_output)

        # NOTE: the predictions are necessary to calcualte the mAP
        if self.current_stage in ("validate", "test"):
            return {"predictions": processed_detections, "one_to_many": x, "one_to_one": one_to_one_output}

        # Inference mode
        return processed_detections

    def _process_filter_predictions(self, features: list[Tensor]) -> Tensor:
        """
        Process feature maps into box coordinates and class scores, then filter to keep top predictions.

        This method performs several key steps:
        1. Concatenates predictions from different feature maps
        2. Updates cell centers and strides if input dimensions have changed
        3. Decodes raw predictions into box coordinates and class probabilities
        4. Filters predictions to keep only the top K detections

        Args:
            features (list[Tensor]): List of 3 prediction tensors from different scales (P3, P4, P5)
                Each tensor shape: (batch_size, channels, height, width)
                where channels = num_classes + 16*4 coordinates
                Example for 640x640 input with 80 classes:
                [
                    Tensor(B, 144, 80, 80),  # P3/8 predictions
                    Tensor(B, 144, 40, 40),  # P4/16 predictions
                    Tensor(B, 144, 20, 20)   # P5/32 predictions
                ]

        Returns:
            Tensor: Filtered and sorted detections of shape (batch_size, detection_limit, 6)
                   where each detection contains:
                   - Box coordinates (x1, y1, x2, y2) in input image scale
                   - Confidence score (0-1)
                   - Class index (0-num_classes)

                   Example for 300 detection limit:
                   Tensor(B, 300, 6) where 6 = [x1, y1, x2, y2, confidence, class_id]

        Note:
            - The method automatically updates cell centers and strides when input dimensions change
            - Only the top K (detection_limit) predictions are kept, sorted by confidence
            - Box coordinates are scaled to match the input image dimensions
        """
        shape = features[0].shape
        predictions = torch.cat([rearrange(f, "b c h w -> b c (h w)") for f in features], dim=2)
        batch_size, _, cell_count = predictions.shape

        # NOTE: this is only called once on the first forward pass
        if self.previous_shape != shape:
            self.cell_centers, self.cell_strides = generate_cell_centers_and_strides(
                feature_maps=features, feature_map_strides=self.feature_map_strides
            )
            self.cell_centers = rearrange(self.cell_centers, "cells dims -> 1 dims cells")
            self.cell_strides = rearrange(self.cell_strides, "s 1 -> 1 s")
            self.previous_shape = shape

        # Split and process boxes and classes
        boxes, classes = predictions.split((self.num_dfl_bins * 4, self.num_classes), dim=1)
        boxes_distributions = self.dfl(boxes)
        boxes_coords_unscaled = distribution_to_box(boxes_distributions, self.cell_centers, dim=1)
        boxes_coords = boxes_coords_unscaled * self.cell_strides
        class_scores = classes.sigmoid()

        boxes_coords = rearrange(boxes_coords, "b coords cells -> b cells coords")
        class_scores = rearrange(class_scores, "b num_classes cells -> b cells num_classes")

        # Select top predictions
        confidence_topk, topk_indices = class_scores.amax(dim=-1).topk(min(self.detection_limit, cell_count))
        topk_indices = rearrange(topk_indices, "b n -> b n 1")

        # Gather corresponding predictions
        filtered_boxes = boxes_coords.gather(dim=1, index=topk_indices.repeat(1, 1, 4))
        filtered_scores = class_scores.gather(dim=1, index=topk_indices.repeat(1, 1, self.num_classes))

        # Get final scores and class indices
        final_scores, class_indices = rearrange(filtered_scores, "b n num_classes -> b (n num_classes)").topk(
            min(self.detection_limit, cell_count)
        )

        batch_indices = rearrange(torch.arange(batch_size, device=boxes_coords.device), "b -> b 1")

        return torch.cat(
            [
                filtered_boxes[batch_indices, class_indices // self.num_classes],
                rearrange(final_scores, "b n -> b n 1"),
                rearrange((class_indices % self.num_classes).float(), "b n -> b n 1"),
            ],
            dim=-1,
        )

    def initialize_biases(self):
        """Initialize network biases for optimal convergence."""

        self._initialize_reg_heads_biases()
        self._initialize_cls_heads_biases()

    def _initialize_reg_heads_biases(self):
        """Initialize biases for regression heads."""

        def reg_head_initialize_biases(reg_head: nn.ModuleList):
            for reg_branch in reg_head:
                # Cell centers initial predictions to 1.0 (avoids extreme coord values early)
                # Aligns with DFL setup and grid cell centering logic
                nn.init.constant_(reg_branch[-1].bias, 1.0)

        reg_head_initialize_biases(self.reg_head)
        reg_head_initialize_biases(self.one_to_one_reg_head)

    def _initialize_cls_heads_biases(self):
        """Initialize biases for classification heads."""

        def cls_head_initialize_biases(cls_head: nn.ModuleList):
            for idx, cls_branch in enumerate(cls_head):
                # Initialize with low confidence
                # Balances: 4=estimated objects/img, 640=reference img size
                # Prevents early overconfidence in background/incorrect classes
                # NOTE: without it the cls loss starts as a huge number
                nn.init.constant_(
                    cls_branch[-1].bias, torch.log(4 / self.num_classes / (640 / self.feature_map_strides[idx]) ** 2)
                )

        cls_head_initialize_biases(self.cls_head)
        cls_head_initialize_biases(self.one_to_one_cls_head)

    def update_num_classes(self, num_classes: int) -> None:
        """
        Updates the detection head for a new number of classes.

        Args:
            num_classes: The new number of classes
        """
        if self.num_classes == num_classes:
            return

        # NOTE: the dlf and reg heads are not re-initialized as they are independent of class count
        # and the cls heads are re-initialized with the new number of classes (with the initial biases)
        self._initialize_cls_heads()
        self._initialize_cls_heads_biases()
