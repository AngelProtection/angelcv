import cv2
import numpy as np


def generate_distinct_colors(num_colors: int) -> list[tuple[int, int, int]]:
    """
    Generate visually distinct colors for class visualization using OpenCV colormaps.

    This function generates deterministic, visually distinct colors using OpenCV's
    built-in colormaps combined with strategic color spacing. It ensures that:
    - The same number of colors always produces the same result
    - Consecutive colors are maximally different (no gradients)
    - All colors are visually distinct

    Args:
        num_colors (int): Number of colors to generate

    Returns:
        list[tuple[int, int, int]]: List of RGB color tuples (0-255 range)
    """
    if num_colors <= 0:
        return []

    colors = []

    # Use OpenCV's HSV colormap for good color distribution
    # Generate more colors than needed to allow for strategic selection
    total_range = max(num_colors * 2, 256)

    # Create a range of hue values distributed across the full spectrum
    hue_values = np.linspace(0, 179, total_range, dtype=np.uint8)

    # Create HSV image with full saturation and value for bright, distinct colors
    hsv_img = np.zeros((1, total_range, 3), dtype=np.uint8)
    hsv_img[0, :, 0] = hue_values  # Hue channel
    hsv_img[0, :, 1] = 255  # Full saturation
    hsv_img[0, :, 2] = 255  # Full value (brightness)

    # Convert to RGB using OpenCV
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    # Extract colors with strategic spacing to maximize visual difference
    # Use golden ratio for optimal distribution
    golden_ratio = 0.618033988749895

    for i in range(num_colors):
        # Use golden ratio spacing to ensure maximum visual separation
        index = int((i * golden_ratio * total_range) % total_range)

        # Extract RGB values
        r, g, b = rgb_img[0, index]

        # Apply slight variations to avoid identical colors and improve distinctness
        # Alternate between slightly different brightness levels
        brightness_factor = 0.85 + (i % 4) * 0.05  # 0.85, 0.90, 0.95, 1.0

        r = int(r * brightness_factor)
        g = int(g * brightness_factor)
        b = int(b * brightness_factor)

        # Ensure values stay in valid range
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        colors.append((r, g, b))

    return colors
