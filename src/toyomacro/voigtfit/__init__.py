# voigtfit - High-performance Voigt Fitting Pipeline
#
# Architecture:
#   Stage 1: Amplitude-only via weight matrix (400M spec/s, memory bandwidth limited)
#   Solvers (for shift/width recovery):
#     - 4-step residual projection: 1.5M spec/s (amp + δE + δσ + amp correction)
#     - 6-step Hessian correction:  0.8M spec/s (+ 2nd order correction)
#     - dict1d (energy shift):      1.0M spec/s (68-entry dictionary)
#     - dict2d (shift + width):     0.8M spec/s (δE × δσ grid)
#     - adaptive (quality-first):   0.4M spec/s (4-step + dict auto-selection)
#
# Performance (8K image = 33M spectra; recorded on Apple M3 Max):
#   Stage 1 fit-only: 33M @ ~400M/s = ~0.08s
#   Roundtrip E2E:    28M spec/s (gen + noise + fit + PSNR)

# MLX capability API (always importable)
from ._mlx_support import mlx_installed, mlx_usable, require_mlx
from .benchmarks.benchmark_h5 import (
    BenchmarkH5,
    BenchmarkH5Config,
    run_unified_benchmark,
)
from .benchmarks.psnr_noise_benchmark import (
    NoiseBenchmarkResult,
    plot_psnr_vs_noise,
    print_benchmark_summary,
    run_psnr_noise_benchmark,
)
from .benchmarks.reconstruction_benchmark import (
    BenchmarkResult,
    ElementConfig,
    ReconstructionBenchmark,
    run_demo_benchmark,
)

# Roundtrip Benchmark (Image -> Spectra -> VoigtFit -> Image)
from .benchmarks.roundtrip_benchmark import (
    NoiseSweepResult,
    RoundtripBenchmark,
    RoundtripResult,
    quick_psnr_check,
    run_gif_roundtrip,
    run_media_roundtrip,
    run_noise_sweep,
    run_roundtrip_benchmark,
)
from .frame_io import (
    ApngWriter,
    FrameMetadata,
    GifWriter,
    ImageSequenceSource,
    ImageSequenceWriter,
    NumpyFrameSource,
    PILFrameSource,
    compose_side_by_side_canvas,
    get_writer,
    open_frames,
)
from .gauss_newton import GaussNewtonRefiner, RefineMode, RefineResult
from .h5io import (
    FitResult as H5FitResult,
)

# I/O and image utilities
from .h5io import (
    XPSData,
    read_fitpara,
    read_project_colors,
    read_project_elements,
    read_spectra,
    write_fitpara,
)
from .image_utils import (
    PSNRResult,
    amplitudes_to_rgb,
    compare_images,
    fitpara_to_rgb,
    get_color_mapping,
    get_demo_color_mapping,
    load_image,
    psnr,
    save_gif,
    save_image,
    save_side_by_side,
    save_side_by_side_gif,
)
from .pipeline import FitResult, HybridPipeline
from .spectra_generator import (
    DEMO_PRESET,
    NOISE_LEVELS,
    ElementPreset,
    ElementSpec,
    GeneratorConfig,
    NoiseConfig,
    SpectraGenerator,
    SpectralConfig,
    add_gaussian_noise,
    add_mixed_noise,
    add_poisson_noise,
    decompose_image_to_amplitudes,
    generate_noise_sweep,
    get_element_preset,
    list_element_presets,
    register_element_preset,
)
from .stage2 import Stage2Config, Stage2Refiner, Stage2Result
from .varpro import VarProFitter
from .voigt_jacobian import voigt_jacobian_batch, voigt_profile, voigt_with_jacobian
from .weight_cache import WeightMatrixCache

# MLX-accelerated modules (optional)
try:
    from .faddeeva_mlx import (
        get_faddeeva_table,
        voigt_basis_mlx,
        voigt_profile_mlx,
        voigt_with_jacobian_mlx,
    )
    from .stage2_mlx import Stage2MLXConfig, Stage2MLXRefiner, Stage2MLXResult
    HAS_MLX = mlx_usable()  # installed AND the default device can execute work
except ImportError:
    HAS_MLX = False

# voigtfit ships as part of the toyomacro distribution and shares its
# version (single source: pyproject.toml, via toyomacro.__version__).
from toyomacro import __version__  # noqa: E402

__all__ = [
    # Pipeline
    "WeightMatrixCache",
    "HybridPipeline",
    "FitResult",
    # Stage 2 (CPU)
    "Stage2Refiner",
    "Stage2Config",
    "Stage2Result",
    # Gauss-Newton
    "GaussNewtonRefiner",
    "RefineMode",
    "RefineResult",
    # VarPro reference fitter (per-spectrum, SciPy; not the batch path)
    "VarProFitter",
    # Voigt functions
    "voigt_profile",
    "voigt_with_jacobian",
    "voigt_jacobian_batch",
    # I/O
    "read_spectra",
    "read_fitpara",
    "write_fitpara",
    "read_project_colors",
    "read_project_elements",
    "XPSData",
    "H5FitResult",
    # Image utilities
    "psnr",
    "compare_images",
    "amplitudes_to_rgb",
    "fitpara_to_rgb",
    "load_image",
    "save_image",
    "get_demo_color_mapping",
    "get_color_mapping",
    "PSNRResult",
    # Benchmark
    "ReconstructionBenchmark",
    "BenchmarkResult",
    "ElementConfig",
    "run_demo_benchmark",
    # Spectra Generation
    "SpectraGenerator",
    "NoiseConfig",
    "GeneratorConfig",
    "ElementSpec",
    "ElementPreset",
    "DEMO_PRESET",
    "get_element_preset",
    "register_element_preset",
    "list_element_presets",
    "add_poisson_noise",
    "add_gaussian_noise",
    "add_mixed_noise",
    "decompose_image_to_amplitudes",
    "generate_noise_sweep",
    # PSNR Noise Benchmark
    "run_psnr_noise_benchmark",
    "plot_psnr_vs_noise",
    "print_benchmark_summary",
    "NoiseBenchmarkResult",
    # Unified Benchmark H5
    "BenchmarkH5",
    "BenchmarkH5Config",
    "run_unified_benchmark",
    # Frame I/O abstractions
    "open_frames",
    "get_writer",
    "compose_side_by_side_canvas",
    "FrameMetadata",
    "PILFrameSource",
    "NumpyFrameSource",
    "ImageSequenceSource",
    "GifWriter",
    "ApngWriter",
    "ImageSequenceWriter",
    # Roundtrip Benchmark (Image -> Spectra -> VoigtFit -> Image)
    "RoundtripBenchmark",
    "RoundtripResult",
    "NoiseSweepResult",
    "run_roundtrip_benchmark",
    "run_noise_sweep",
    "quick_psnr_check",
    "run_media_roundtrip",
    "run_gif_roundtrip",
    "save_gif",
    "save_side_by_side",
    "save_side_by_side_gif",
    "NOISE_LEVELS",
    "SpectralConfig",
    # MLX availability flag + capability API
    "HAS_MLX",
    "mlx_installed",
    "mlx_usable",
    "require_mlx",
]

# Add MLX exports if available
if HAS_MLX:
    __all__.extend([
        "voigt_profile_mlx",
        "voigt_basis_mlx",
        "voigt_with_jacobian_mlx",
        "get_faddeeva_table",
        "Stage2MLXRefiner",
        "Stage2MLXConfig",
        "Stage2MLXResult",
    ])
