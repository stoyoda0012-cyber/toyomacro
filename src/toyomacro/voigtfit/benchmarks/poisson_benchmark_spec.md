# Poisson Noise Generation Benchmark Specification

**Created**: 2026-02-16 (dev-log 22)
**Status**: Next session (dev-log 24) task

## Purpose

GPU (MLX) Poisson noise generation: compare throughput and distribution accuracy across 3 methods.

## Background

XPS spectral analysis (DepthInference project) requires Monte Carlo generation of massive Poisson-noised spectra.
Evaluating Apple Silicon + MLX acceleration: quantify accuracy vs throughput tradeoff of Gaussian approximation.

### Theory

- **Box-Muller**: 2 uniform RNG -> 2 normal RNG. Fixed 2-step, branch-free -> GPU optimal
- **Cornish-Fisher**: Add skewness correction `gamma1*(z^2-1)/6` to Gaussian z. Extra cost: a few multiplies
- **Exact Poisson (Knuth)**: Loop lambda times -> GPU-hostile. PTRD also rejection-branching -> GPU-hostile
- **Gamma approximation**: Rejection sampling, warp thread divergence -> intermediate speed

## Environment

- Python 3.11+, MLX (Apple Silicon GPU), NumPy, matplotlib, scipy

## 3 Methods to Implement

### Method 1: Box-Muller + Gaussian Approximation (MLX)

```python
def poisson_gaussian(lam, key):
    z = mx.random.normal(lam.shape, key=key)
    return mx.maximum(lam + mx.sqrt(mx.maximum(lam, 1.0)) * z, 0.0)
```

### Method 2: Box-Muller + Cornish-Fisher Correction (MLX)

```python
def poisson_cornish_fisher(lam, key):
    z = mx.random.normal(lam.shape, key=key)
    gamma1 = 1.0 / mx.sqrt(mx.maximum(lam, 1.0))
    z_corrected = z + gamma1 * (z * z - 1.0) / 6.0
    return mx.maximum(lam + mx.sqrt(mx.maximum(lam, 1.0)) * z_corrected, 0.0)
```

### Method 3: NumPy Exact Poisson (CPU, ground truth)

```python
def poisson_exact(lam):
    return np.random.poisson(lam)
```

## Benchmark Specification

### Parameters

| Parameter | Value |
|-----------|-------|
| Lambda values | [1, 5, 10, 25, 50, 100, 500, 1000] |
| Throughput samples | 1,000,000 per lambda |
| Distribution samples | 10,000,000 per lambda |
| Throughput repeats | 10 runs, take median |

### Throughput Measurement

- Unit: M samples/sec
- MLX methods: measure until `mx.eval()` completion (beware lazy evaluation)
- Warmup: 2 dry runs per method before measurement

### Distribution Accuracy Verification

For each lambda x each method, compute and compare against exact Poisson:

1. **Mean** (expected: lambda)
2. **Variance** (expected: lambda)
3. **Skewness** (expected: 1/sqrt(lambda))
4. **Kurtosis** (expected: 1/lambda)
5. **P(X < 0) ratio** (Gaussian methods only, measure pre-clamp values)

## Output

### 1. Console Output

Table format for each method x each lambda (throughput table + distribution accuracy table)

### 2. Plot (single PNG, 4-panel layout)

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
```

| Panel | Content |
|-------|---------|
| Top-left | Throughput vs lambda (3 methods, log scale) |
| Top-right | Skewness error vs lambda (delta from theory, 3 methods) |
| Bottom-left | Histogram overlay at lambda=5 (3 methods + theoretical PMF, bins=range(0, 25)) |
| Bottom-right | Histogram overlay at lambda=50 (3 methods + theoretical PMF, bins=range(20, 80)) |

Save: `poisson_benchmark_results.png` (dpi=150)

## File Structure

Single file: `poisson_benchmark.py`. Run: `python poisson_benchmark.py`

## Implementation Notes

- MLX lazy evaluation: always call `mx.eval()` within measurement interval
- Collect statistics both pre-clamp and post-clamp for comparison
- Graceful exit with error message if MLX not installed
- Fixed random seeds: `mx.random.key(42)`, `np.random.seed(42)`
- Japanese comments for readability
