"""
Image utilities for XPS mapping visualization and quality assessment.

Implements MATLAB-compatible RGB image generation and PSNR calculation
for evaluating reconstruction quality.

The normalization method matches Toyomacro's PresentMap_Sub.m:
    Map(:,:,ch) = uint8(255 * channel / max(channel))
Date: 2026-01-22
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PSNRResult:
    """Container for PSNR comparison results."""
    psnr_total: float
    psnr_r: float
    psnr_g: float
    psnr_b: float
    mean_abs_error: float
    max_diff: int
    ssim: float | None = None

    def __str__(self) -> str:
        return (
            f"PSNR: {self.psnr_total:.2f} dB (R:{self.psnr_r:.2f}, "
            f"G:{self.psnr_g:.2f}, B:{self.psnr_b:.2f}), "
            f"MAE: {self.mean_abs_error:.2f}"
        )


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: int = 255) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.

    Args:
        img1: First image (uint8 or float)
        img2: Second image (uint8 or float)
        max_val: Maximum pixel value (255 for uint8)

    Returns:
        PSNR in dB (inf if images are identical)
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))


def compare_images(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: int = 255,
) -> PSNRResult:
    """
    Comprehensive image quality comparison.

    Args:
        img1: Reference image (H, W, 3) uint8
        img2: Test image (H, W, 3) uint8
        max_val: Maximum pixel value

    Returns:
        PSNRResult with total and per-channel metrics
    """
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"

    # Total PSNR
    psnr_total = psnr(img1, img2, max_val)

    # Per-channel PSNR
    psnr_r = psnr(img1[:, :, 0], img2[:, :, 0], max_val)
    psnr_g = psnr(img1[:, :, 1], img2[:, :, 1], max_val)
    psnr_b = psnr(img1[:, :, 2], img2[:, :, 2], max_val)

    # Error statistics
    diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
    mean_abs_error = diff.mean()
    max_diff = diff.max()

    return PSNRResult(
        psnr_total=psnr_total,
        psnr_r=psnr_r,
        psnr_g=psnr_g,
        psnr_b=psnr_b,
        mean_abs_error=mean_abs_error,
        max_diff=int(max_diff),
    )


def amplitudes_to_rgb(
    amplitudes: dict[str, np.ndarray],
    color_mapping: np.ndarray,
    component_order: list[tuple[str, int]],
    image_shape: tuple[int, int],
    reshape_order: str = 'F',
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Convert component amplitudes to RGB image using MATLAB normalization.

    This implements the same algorithm as PresentMap_Sub.m (line 149-150):
        r = reshape(sum(r,2),y,x);
        Map(:,:,1) = uint8(255*r./max(r,[],'all'));

    Args:
        amplitudes: Dict mapping file_key to amplitude array (n_comp, n_spectra)
        color_mapping: (3, n_components) RGB weights from ezdeprof/Col
        component_order: List of (file_key, comp_idx) in color_mapping order
        image_shape: (height, width) of output image
        reshape_order: 'F' for column-major (MATLAB), 'C' for row-major
        clip_negative: If True, clip negative amplitudes to 0

    Returns:
        RGB image as (height, width, 3) uint8 array
    """
    height, width = image_shape
    n_spectra = height * width

    # Initialize RGB channels
    r_channel = np.zeros(n_spectra, dtype=np.float64)
    g_channel = np.zeros(n_spectra, dtype=np.float64)
    b_channel = np.zeros(n_spectra, dtype=np.float64)

    # Accumulate weighted amplitudes
    for k, (file_key, comp_idx) in enumerate(component_order):
        amp = amplitudes[file_key][comp_idx, :].astype(np.float64)

        if clip_negative:
            amp = np.maximum(amp, 0)

        r_channel += amp * color_mapping[0, k]
        g_channel += amp * color_mapping[1, k]
        b_channel += amp * color_mapping[2, k]

    # Reshape to image (column-major for MATLAB compatibility)
    r_img = r_channel.reshape(height, width, order=reshape_order)
    g_img = g_channel.reshape(height, width, order=reshape_order)
    b_img = b_channel.reshape(height, width, order=reshape_order)

    # Normalize each channel by its max (MATLAB method)
    r_max = np.max(r_img)
    g_max = np.max(g_img)
    b_max = np.max(b_img)

    r_norm = r_img / r_max if r_max > 0 else r_img
    g_norm = g_img / g_max if g_max > 0 else g_img
    b_norm = b_img / b_max if b_max > 0 else b_img

    # Combine and convert to uint8
    rgb = np.stack([r_norm, g_norm, b_norm], axis=2)
    rgb_uint8 = (255 * np.clip(rgb, 0, 1)).astype(np.uint8)

    return rgb_uint8


def fitpara_to_rgb(
    fitpara_dict: dict[str, np.ndarray],
    color_mapping: np.ndarray,
    component_order: list[tuple[str, int]],
    image_shape: tuple[int, int],
    amplitude_index: int = 0,
    reshape_order: str = 'F',
) -> np.ndarray:
    """
    Convert fitpara arrays to RGB image.

    Args:
        fitpara_dict: Dict mapping file_key to fitpara array (n_spectra, n_comp, 9)
        color_mapping: (3, n_components) RGB weights
        component_order: List of (file_key, comp_idx) in color_mapping order
        image_shape: (height, width) of output image
        amplitude_index: Index of amplitude in fitpara (default: 0)
        reshape_order: 'F' for column-major (MATLAB)

    Returns:
        RGB image as (height, width, 3) uint8 array
    """
    # Extract amplitudes from fitpara
    amplitudes = {}
    for file_key, fitpara in fitpara_dict.items():
        # fitpara: (n_spectra, n_comp, 9) -> amplitudes: (n_comp, n_spectra)
        n_spectra, n_comp, _ = fitpara.shape
        amp = np.zeros((n_comp, n_spectra), dtype=np.float32)
        for j in range(n_comp):
            amp[j, :] = fitpara[:, j, amplitude_index]
        amplitudes[file_key] = amp

    return amplitudes_to_rgb(
        amplitudes=amplitudes,
        color_mapping=color_mapping,
        component_order=component_order,
        image_shape=image_shape,
        reshape_order=reshape_order,
    )


def load_image(
    path: str | Path,
    all_frames: bool = False,
) -> np.ndarray:
    """
    Load image from file. Supports static images, animated GIFs, and more.

    Delegates to frame_io.open_frames() for format-agnostic frame loading.

    Args:
        path: Path to image file (PNG, JPG, GIF, etc.), directory of images,
              or video file (requires imageio[pyav])
        all_frames: If True and the source has multiple frames, return all frames
                    as (n_frames, height, width, 3). Otherwise return the first
                    frame as (height, width, 3).

    Returns:
        Static image or single frame: (height, width, 3) uint8 array
        Multi-frame with all_frames=True: (n_frames, height, width, 3) uint8 array
    """
    from .frame_io import open_frames
    source = open_frames(path)
    if all_frames and source.n_frames > 1:
        return source.load_all()
    return source[0]


def save_image(
    image: np.ndarray,
    path: str | Path,
    quality: int = 95,
) -> None:
    """
    Save image to file.

    Args:
        image: (height, width, 3) uint8 array
        path: Output path
        quality: JPEG quality (ignored for PNG)
    """
    from PIL import Image
    path = Path(path)
    img = Image.fromarray(image)

    if path.suffix.lower() in ['.jpg', '.jpeg']:
        img.save(path, quality=quality)
    else:
        img.save(path)


def save_gif(
    frames: np.ndarray,
    path: str | Path,
    duration: int = 50,
    loop: int = 0,
) -> None:
    """
    Save animated GIF from numpy frames.

    Args:
        frames: (n_frames, height, width, 3) uint8 array
        path: Output path (.gif)
        duration: Duration per frame in milliseconds (default: 50ms = 20fps)
        loop: 0 for infinite loop, N to loop N times
    """
    from PIL import Image
    path = Path(path)

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=loop,
        optimize=False,
    )


def save_side_by_side(
    frame_groups: list[np.ndarray],
    labels: list[str] | None,
    path: str | Path,
    duration: int = 50,
    loop: int = 0,
    label_height: int = 20,
    gap: int = 4,
    gt_palette_budget: int = 192,
    psnr_per_frame: list[list[float] | None] | None = None,
) -> None:
    """
    Save animated image with multiple frame sequences side by side.

    Auto-detects output format from file extension:
      .gif  → GIF with global palette quantization (no flicker)
      .apng/.png → APNG with full 24-bit RGB
      .mp4/.webm → Video (requires imageio[pyav])
      directory → Numbered PNGs

    Args:
        frame_groups: List of (n_frames, H, W, 3) uint8 arrays.
                      All must have the same n_frames, H, W.
        labels: Optional labels for each column (e.g., ['Ground Truth', 'Noisy'])
        path: Output path (.gif, .apng/.png, .mp4, or directory)
        duration: Duration per frame in ms
        loop: 0 for infinite loop
        label_height: Height of label bar in pixels (0 to disable)
        gap: Gap between columns in pixels
        gt_palette_budget: Number of palette colors reserved for GT column (GIF only)
        psnr_per_frame: Per-column, per-frame PSNR values to display in label area.
                        List of length n_cols. Each element is either None (no PSNR,
                        e.g. for GT column) or a list of n_frames float values.
                        Example: [None, [31.2, 30.8, ...], [24.5, 24.1, ...]]
    """
    from .frame_io import FrameMetadata, GifWriter, get_writer

    path = Path(path)
    metadata = FrameMetadata(fps=1000.0 / max(duration, 1))

    writer = get_writer(path)

    # Pass gt_palette_budget to GifWriter if applicable
    if isinstance(writer, GifWriter):
        writer.gt_palette_budget = gt_palette_budget

    writer.write_sequence(
        frame_groups=frame_groups,
        labels=labels,
        path=path,
        metadata=metadata,
        psnr_per_frame=psnr_per_frame,
        label_height=label_height,
        gap=gap,
    )


def save_side_by_side_gif(
    frame_groups: list[np.ndarray],
    labels: list[str] | None,
    path: str | Path,
    duration: int = 50,
    loop: int = 0,
    label_height: int = 20,
    gap: int = 4,
    gt_palette_budget: int = 192,
    psnr_per_frame: list[list[float] | None] | None = None,
) -> None:
    """Backward-compatible alias for save_side_by_side().

    See save_side_by_side() for full documentation.
    """
    save_side_by_side(
        frame_groups=frame_groups,
        labels=labels,
        path=path,
        duration=duration,
        loop=loop,
        label_height=label_height,
        gap=gap,
        gt_palette_budget=gt_palette_budget,
        psnr_per_frame=psnr_per_frame,
    )


def get_color_mapping(preset_name: str = 'demo') -> tuple[np.ndarray, list[tuple[str, int]]]:
    """
    Get color mapping and component order for a named element preset.

    Args:
        preset_name: Registered preset name (default: 'demo')

    Returns:
        (color_mapping, component_order) tuple
            color_mapping: (3, n_components) RGB weights
            component_order: List of (file_key, comp_idx_in_file)
    """
    from .spectra_generator import get_element_preset
    preset = get_element_preset(preset_name)
    return preset.color_mapping.copy(), list(preset.component_order)


def get_demo_color_mapping() -> tuple[np.ndarray, list[tuple[str, int]]]:
    """
    Get color mapping for the 6-component demo preset.

    Backward-compatible wrapper around get_color_mapping('demo').

    Returns:
        (color_mapping, component_order) tuple
    """
    return get_color_mapping('demo')


def print_comparison_report(
    results: dict[str, PSNRResult],
    title: str = "Image Quality Comparison",
) -> None:
    """
    Print formatted comparison report.

    Args:
        results: Dict mapping comparison name to PSNRResult
        title: Report title
    """
    print("=" * 70)
    print(title)
    print("=" * 70)

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  PSNR (total): {result.psnr_total:.2f} dB")
        print(f"  PSNR (R/G/B): {result.psnr_r:.2f} / {result.psnr_g:.2f} / {result.psnr_b:.2f} dB")
        print(f"  Mean abs error: {result.mean_abs_error:.2f}")
        print(f"  Max diff: {result.max_diff}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("  > 40 dB: Excellent (nearly identical)")
    print("  30-40 dB: Good (minor differences)")
    print("  20-30 dB: Fair (visible differences)")
    print("  < 20 dB: Poor (significant differences)")
    print("=" * 70)
