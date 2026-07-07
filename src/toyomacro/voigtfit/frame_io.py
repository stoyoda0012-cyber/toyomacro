"""
Frame I/O abstraction for multi-format media support.

Provides format-agnostic FrameSource (input) and FrameWriter (output) interfaces,
decoupling the VoigtFit roundtrip pipeline from specific image/video formats.

Supported input formats:
  - GIF, APNG, PNG, JPEG, TIFF, BMP (via PIL)
  - Directory of numbered images (ImageSequenceSource)
  - Pre-loaded numpy arrays (NumpyFrameSource)
  - MP4/WebM (planned, via imageio[pyav])

Supported output formats:
  - GIF with global palette quantization (GifWriter)
  - APNG with full 24-bit RGB (ApngWriter)
  - Directory of numbered PNGs (ImageSequenceWriter)
  - MP4/WebM (planned, via imageio[pyav])
Date: 2026-02-18
"""

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Protocol,
    runtime_checkable,
)

import numpy as np

# ============================================================================
# Metadata
# ============================================================================

@dataclass
class FrameMetadata:
    """Timing and format metadata for frame sequences."""
    fps: float = 20.0
    frame_durations_ms: list[int] | None = None  # per-frame (GIF)

    @property
    def duration_ms(self) -> int:
        """Uniform frame duration in ms (for output)."""
        return int(round(1000.0 / self.fps))


# ============================================================================
# FrameSource Protocol + Implementations
# ============================================================================

@runtime_checkable
class FrameSource(Protocol):
    """Protocol for reading frame sequences from any format."""

    @property
    def n_frames(self) -> int: ...

    @property
    def frame_shape(self) -> tuple[int, int]: ...  # (height, width)

    @property
    def metadata(self) -> FrameMetadata: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> np.ndarray: ...  # (H, W, 3) uint8

    def __iter__(self) -> Iterator[np.ndarray]: ...

    def load_all(self) -> np.ndarray: ...  # (n_frames, H, W, 3) uint8


class PILFrameSource:
    """Reads GIF, APNG, static images via PIL.

    Supports all formats that PIL/Pillow can open. Multi-frame formats
    (GIF, APNG) are accessed via seek(). Static images have n_frames=1.
    """

    def __init__(self, path: str | Path):
        from PIL import Image
        self._path = Path(path)
        self._img = Image.open(self._path)
        self._n_frames = getattr(self._img, 'n_frames', 1)

        # Read first frame for dimensions
        first = self._img.convert('RGB')
        self._width, self._height = first.size

        # Extract GIF frame durations
        durations = []
        if self._n_frames > 1:
            for i in range(self._n_frames):
                self._img.seek(i)
                durations.append(self._img.info.get('duration', 50))
        avg_duration = np.mean(durations) if durations else 50
        self._metadata = FrameMetadata(
            fps=1000.0 / max(avg_duration, 1),
            frame_durations_ms=durations if durations else None,
        )

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self._height, self._width)

    @property
    def metadata(self) -> FrameMetadata:
        return self._metadata

    def __len__(self) -> int:
        return self._n_frames

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0:
            index += self._n_frames
        if index < 0 or index >= self._n_frames:
            raise IndexError(f"Frame index {index} out of range [0, {self._n_frames})")
        self._img.seek(index)
        return np.array(self._img.convert('RGB'), dtype=np.uint8)

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(self._n_frames):
            yield self[i]

    def load_all(self) -> np.ndarray:
        return np.stack(list(self), axis=0)

    def __repr__(self) -> str:
        return (f"PILFrameSource('{self._path.name}', "
                f"{self._n_frames} frames, {self._height}x{self._width})")


class NumpyFrameSource:
    """Wraps an existing numpy array as a FrameSource.

    Useful for testing and when frames are already loaded in memory.
    """

    def __init__(
        self,
        frames: np.ndarray,
        fps: float = 20.0,
    ):
        """
        Args:
            frames: (n_frames, H, W, 3) or (H, W, 3) uint8 array
            fps: Frames per second for metadata
        """
        if frames.ndim == 3:
            frames = frames[np.newaxis]
        if frames.ndim != 4 or frames.shape[3] != 3:
            raise ValueError(
                f"Expected (n_frames, H, W, 3) or (H, W, 3), got {frames.shape}")
        self._frames = frames.astype(np.uint8)
        self._metadata = FrameMetadata(fps=fps)

    @property
    def n_frames(self) -> int:
        return self._frames.shape[0]

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self._frames.shape[1], self._frames.shape[2])

    @property
    def metadata(self) -> FrameMetadata:
        return self._metadata

    def __len__(self) -> int:
        return self._frames.shape[0]

    def __getitem__(self, index: int) -> np.ndarray:
        return self._frames[index]

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self._frames[i]

    def load_all(self) -> np.ndarray:
        return self._frames.copy()

    def __repr__(self) -> str:
        h, w = self.frame_shape
        return (f"NumpyFrameSource({self.n_frames} frames, {h}x{w})")


class ImageSequenceSource:
    """Reads a directory of numbered images as a frame sequence.

    Files are sorted by name. Supports any image format PIL can read.
    """

    def __init__(
        self,
        directory: str | Path,
        pattern: str = '*.png',
        fps: float = 20.0,
    ):
        """
        Args:
            directory: Path to directory containing image files
            pattern: Glob pattern to match files (default: '*.png')
            fps: Frames per second for metadata
        """
        from PIL import Image
        self._dir = Path(directory)
        self._files = sorted(self._dir.glob(pattern))
        if not self._files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {directory}")

        # Read first frame for dimensions
        first = Image.open(self._files[0]).convert('RGB')
        self._width, self._height = first.size
        self._metadata = FrameMetadata(fps=fps)

    @property
    def n_frames(self) -> int:
        return len(self._files)

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self._height, self._width)

    @property
    def metadata(self) -> FrameMetadata:
        return self._metadata

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int) -> np.ndarray:
        from PIL import Image
        if index < 0:
            index += len(self._files)
        if index < 0 or index >= len(self._files):
            raise IndexError(f"Frame index {index} out of range [0, {len(self._files)})")
        return np.array(
            Image.open(self._files[index]).convert('RGB'),
            dtype=np.uint8,
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        from PIL import Image
        for f in self._files:
            yield np.array(Image.open(f).convert('RGB'), dtype=np.uint8)

    def load_all(self) -> np.ndarray:
        return np.stack(list(self), axis=0)

    def __repr__(self) -> str:
        return (f"ImageSequenceSource('{self._dir.name}/', "
                f"{len(self._files)} frames, {self._height}x{self._width})")


# ============================================================================
# Factory: open_frames
# ============================================================================

def open_frames(
    path: str | Path,
    fps: float = 20.0,
) -> FrameSource:
    """Auto-detect format and return appropriate FrameSource.

    Supports:
      - .gif, .apng, .png, .jpg, .jpeg, .tiff, .bmp → PILFrameSource
      - Directory path → ImageSequenceSource
      - .mp4, .webm, .avi, .mov → VideoFrameSource (requires imageio[pyav])

    Args:
        path: Path to image file, animated GIF, video, or image directory
        fps: Default frames per second (used when format doesn't specify)

    Returns:
        FrameSource instance
    """
    path = Path(path)

    if path.is_dir():
        # Try common image extensions
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp'):
            files = sorted(path.glob(ext))
            if files:
                return ImageSequenceSource(path, pattern=ext, fps=fps)
        raise FileNotFoundError(
            f"No image files found in directory: {path}")

    suffix = path.suffix.lower()

    if suffix in ('.mp4', '.webm', '.avi', '.mov', '.mkv'):
        # Video format — defer to VideoFrameSource (future)
        try:
            from .frame_io_video import VideoFrameSource
            return VideoFrameSource(path)
        except ImportError:
            raise ImportError(
                f"Video format '{suffix}' requires imageio[pyav]. "
                "Install with: pip install 'imageio[pyav]'"
            )

    # Default: PIL for all image formats
    return PILFrameSource(path)


# ============================================================================
# Canvas Composition (format-agnostic)
# ============================================================================

def compose_side_by_side_canvas(
    frame_groups: list[np.ndarray],
    labels: list[str] | None,
    frame_index: int,
    label_height: int = 20,
    gap: int = 4,
    psnr_per_frame: list[list[float] | None] | None = None,
    font: object | None = None,
) -> np.ndarray:
    """Compose a single canvas frame from multiple frame group columns.

    Format-agnostic: just geometry, label drawing, and PSNR text.
    Does NOT perform any palette quantization or format-specific encoding.

    Args:
        frame_groups: List of (n_frames, H, W, 3) uint8 arrays
        labels: Optional label strings for each column
        frame_index: Which frame to compose
        label_height: Height of label bar in pixels (0 to disable)
        gap: Gap between columns in pixels
        psnr_per_frame: Per-column, per-frame PSNR values
        font: PIL ImageFont to use (if None, auto-load)

    Returns:
        (total_h, total_w, 3) uint8 canvas
    """
    from PIL import Image, ImageDraw

    n_cols = len(frame_groups)
    h, w = frame_groups[0].shape[1], frame_groups[0].shape[2]

    # Auto-scale label height
    if label_height > 0 and label_height <= 20:
        label_height = max(20, int(h * 0.067))

    has_labels = labels is not None and label_height > 0
    total_w = n_cols * w + (n_cols - 1) * gap
    total_h = h + (label_height if has_labels else 0)

    # Build canvas
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    y_offset = label_height if has_labels else 0
    for col, group in enumerate(frame_groups):
        x_offset = col * (w + gap)
        canvas[y_offset:y_offset + h, x_offset:x_offset + w, :] = group[frame_index]

    # Draw labels
    if has_labels and labels:
        if font is None:
            font = _load_font(label_height)
        pil_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)
        for col, label in enumerate(labels):
            x_center = col * (w + gap) + w // 2
            if (psnr_per_frame is not None
                    and col < len(psnr_per_frame)
                    and psnr_per_frame[col] is not None):
                psnr_val = psnr_per_frame[col][frame_index]
                text = f'{label} ({psnr_val:.1f} dB)'
            else:
                text = label
            draw.text(
                (x_center, label_height // 2),
                text,
                fill=(255, 255, 255),
                anchor='mm',
                font=font,
            )
        canvas = np.array(pil_img)

    return canvas


def _load_font(label_height: int):
    """Load a scalable font, falling back to default."""
    from PIL import ImageFont
    font_size = max(12, label_height - 8)
    try:
        return ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except OSError:
            return ImageFont.load_default()


# ============================================================================
# FrameWriter Protocol + Implementations
# ============================================================================

@runtime_checkable
class FrameWriter(Protocol):
    """Protocol for writing frame sequences to various formats."""

    def write_sequence(
        self,
        frame_groups: list[np.ndarray],
        labels: list[str] | None,
        path: str | Path,
        metadata: FrameMetadata,
        psnr_per_frame: list[list[float] | None] | None = None,
        label_height: int = 20,
        gap: int = 4,
    ) -> None: ...


class GifWriter:
    """GIF output with global palette quantization.

    Uses a 2-tier palette strategy (GT + Recon sub-palettes) to minimize
    inter-frame flicker while preserving color fidelity in both GT and
    reconstructed columns.
    """

    def __init__(self, gt_palette_budget: int = 192):
        """
        Args:
            gt_palette_budget: Number of palette colors reserved for
                GT column (first column). Remaining goes to recon columns.
        """
        self.gt_palette_budget = gt_palette_budget

    def write_sequence(
        self,
        frame_groups: list[np.ndarray],
        labels: list[str] | None,
        path: str | Path,
        metadata: FrameMetadata,
        psnr_per_frame: list[list[float] | None] | None = None,
        label_height: int = 20,
        gap: int = 4,
    ) -> None:
        from PIL import Image

        path = Path(path)
        n_cols = len(frame_groups)
        n_frames = frame_groups[0].shape[0]
        h, w = frame_groups[0].shape[1], frame_groups[0].shape[2]

        # Pre-compute font once
        effective_label_height = label_height
        if effective_label_height > 0 and effective_label_height <= 20:
            effective_label_height = max(20, int(h * 0.067))
        font = _load_font(effective_label_height) if labels else None

        # Compute global palettes
        global_gt_pal, global_recon_pal = self._compute_global_palettes(
            frame_groups, n_frames)
        n_gt = len(global_gt_pal)

        # Build combined palette (same for ALL frames)
        if n_cols == 1:
            pal_colors = global_gt_pal
        else:
            pal_colors = np.concatenate(
                [global_gt_pal, global_recon_pal], axis=0)

        # Pad to 256
        if len(pal_colors) < 256:
            pal_colors = np.concatenate([
                pal_colors,
                np.zeros((256 - len(pal_colors), 3), dtype=np.uint8)
            ], axis=0)

        # Frame palette image (reused)
        frame_pal = Image.new('P', (1, 1))
        frame_pal.putpalette(
            list(pal_colors[:256].flatten().astype(int)))

        # GT sub-palette
        gt_sub_pal = Image.new('P', (1, 1))
        gt_pal_padded = np.zeros((256, 3), dtype=np.uint8)
        gt_pal_padded[:len(global_gt_pal)] = global_gt_pal
        gt_sub_pal.putpalette(list(gt_pal_padded.flatten().astype(int)))

        # Recon sub-palette
        recon_sub_pal = None
        if n_cols > 1 and global_recon_pal is not None:
            recon_sub_pal = Image.new('P', (1, 1))
            recon_pal_padded = np.zeros((256, 3), dtype=np.uint8)
            recon_pal_padded[:len(global_recon_pal)] = global_recon_pal
            recon_sub_pal.putpalette(
                list(recon_pal_padded.flatten().astype(int)))

        # Quantize each frame
        has_labels = labels is not None and label_height > 0
        y_offset = effective_label_height if has_labels else 0

        pil_frames = []
        for i in range(n_frames):
            canvas = compose_side_by_side_canvas(
                frame_groups, labels, i,
                label_height=label_height, gap=gap,
                psnr_per_frame=psnr_per_frame, font=font)
            pil_img = Image.fromarray(canvas)

            # Quantize full canvas against combined palette
            q_data = np.array(
                pil_img.quantize(palette=frame_pal,
                                 dither=Image.Dither.NONE))

            # Re-quantize GT column against GT sub-palette
            gt_pil = Image.fromarray(frame_groups[0][i])
            gt_q = gt_pil.quantize(palette=gt_sub_pal,
                                   dither=Image.Dither.NONE)
            q_data[y_offset:y_offset + h, :w] = np.array(gt_q)

            # Re-quantize Recon columns with offset indices
            if n_cols > 1 and recon_sub_pal is not None:
                for col_idx in range(1, n_cols):
                    x_off = col_idx * (w + gap)
                    col_pil = Image.fromarray(frame_groups[col_idx][i])
                    col_q = col_pil.quantize(
                        palette=recon_sub_pal,
                        dither=Image.Dither.NONE)
                    q_data[y_offset:y_offset + h,
                           x_off:x_off + w] = np.array(col_q) + n_gt

            pil_quantized = Image.fromarray(q_data, mode='P')
            pil_quantized.putpalette(
                list(pal_colors[:256].flatten().astype(int)))
            pil_frames.append(pil_quantized)

        # Save
        duration = metadata.frame_durations_ms or metadata.duration_ms
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )

    def _compute_global_palettes(
        self,
        frame_groups: list[np.ndarray],
        n_frames: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute global GT and Recon palettes from all frames.

        Returns:
            (gt_palette, recon_palette) — each is (N, 3) uint8 or None
        """
        from PIL import Image

        n_cols = len(frame_groups)

        # GT palette
        all_gt_px = []
        for i in range(n_frames):
            all_gt_px.append(frame_groups[0][i].reshape(-1, 3))
        all_gt_px = np.concatenate(all_gt_px, axis=0)
        if len(all_gt_px) > 500_000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(all_gt_px), 500_000, replace=False)
            all_gt_px = all_gt_px[idx]

        n_gt_actual = 256 if n_cols == 1 else self.gt_palette_budget
        gt_img = Image.fromarray(
            all_gt_px.reshape(1, -1, 3)[:, :256 * 256, :])
        gt_q = gt_img.quantize(
            colors=n_gt_actual, method=Image.Quantize.MEDIANCUT)
        global_gt_pal = np.array(
            gt_q.getpalette()[:n_gt_actual * 3], dtype=np.uint8
        ).reshape(-1, 3)

        # Recon palette
        global_recon_pal = None
        if n_cols > 1:
            n_recon_budget = 256 - len(global_gt_pal)
            all_recon_px = []
            for i in range(n_frames):
                for col_idx in range(1, n_cols):
                    all_recon_px.append(
                        frame_groups[col_idx][i].reshape(-1, 3))
            all_recon_px = np.concatenate(all_recon_px, axis=0)
            if len(all_recon_px) > 500_000:
                rng = np.random.default_rng(43)
                idx = rng.choice(len(all_recon_px), 500_000, replace=False)
                all_recon_px = all_recon_px[idx]
            recon_img = Image.fromarray(
                all_recon_px.reshape(1, -1, 3)[:, :256 * 256, :])
            recon_q = recon_img.quantize(
                colors=n_recon_budget, method=Image.Quantize.MEDIANCUT)
            global_recon_pal = np.array(
                recon_q.getpalette()[:n_recon_budget * 3], dtype=np.uint8
            ).reshape(-1, 3)

        return global_gt_pal, global_recon_pal


class ApngWriter:
    """APNG output with full 24-bit RGB (no palette quantization)."""

    def write_sequence(
        self,
        frame_groups: list[np.ndarray],
        labels: list[str] | None,
        path: str | Path,
        metadata: FrameMetadata,
        psnr_per_frame: list[list[float] | None] | None = None,
        label_height: int = 20,
        gap: int = 4,
    ) -> None:
        from PIL import Image

        path = Path(path)
        n_frames = frame_groups[0].shape[0]
        h = frame_groups[0].shape[1]

        font = _load_font(max(20, int(h * 0.067))) if labels else None

        pil_frames = []
        for i in range(n_frames):
            canvas = compose_side_by_side_canvas(
                frame_groups, labels, i,
                label_height=label_height, gap=gap,
                psnr_per_frame=psnr_per_frame, font=font)
            pil_frames.append(Image.fromarray(canvas))

        duration = metadata.frame_durations_ms or metadata.duration_ms
        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )


class ImageSequenceWriter:
    """Write individual frames as numbered PNGs to a directory."""

    def write_sequence(
        self,
        frame_groups: list[np.ndarray],
        labels: list[str] | None,
        path: str | Path,
        metadata: FrameMetadata,
        psnr_per_frame: list[list[float] | None] | None = None,
        label_height: int = 20,
        gap: int = 4,
    ) -> None:
        from PIL import Image

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        n_frames = frame_groups[0].shape[0]
        h = frame_groups[0].shape[1]
        font = _load_font(max(20, int(h * 0.067))) if labels else None

        n_digits = len(str(n_frames))
        for i in range(n_frames):
            canvas = compose_side_by_side_canvas(
                frame_groups, labels, i,
                label_height=label_height, gap=gap,
                psnr_per_frame=psnr_per_frame, font=font)
            frame_path = path / f"frame_{i:0{n_digits}d}.png"
            Image.fromarray(canvas).save(frame_path)


# ============================================================================
# Factory: get_writer
# ============================================================================

def get_writer(path: str | Path) -> FrameWriter:
    """Auto-detect writer from file extension.

    Args:
        path: Output file path. Extension determines format:
              .gif → GifWriter
              .apng, .png → ApngWriter
              .mp4, .webm → VideoWriter (future)
              directory / no extension → ImageSequenceWriter

    Returns:
        FrameWriter instance
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.gif':
        return GifWriter()
    elif suffix in ('.apng', '.png'):
        return ApngWriter()
    elif suffix in ('.mp4', '.webm', '.avi', '.mov'):
        try:
            from .frame_io_video import VideoWriter
            return VideoWriter()
        except ImportError:
            raise ImportError(
                f"Video format '{suffix}' requires imageio[pyav]. "
                "Install with: pip install 'imageio[pyav]'"
            )
    elif suffix == '' or path.is_dir():
        return ImageSequenceWriter()
    else:
        # Default to GIF
        warnings.warn(
            f"Unknown output format '{suffix}', defaulting to GIF writer")
        return GifWriter()
