#!/usr/bin/env python3
"""Poisson Noise Generation Benchmark: 3方式比較 (Gaussian / Cornish-Fisher / Exact)

Box-Muller Gaussian近似 (MLX GPU), Cornish-Fisher補正 (MLX GPU),
厳密Poisson (NumPy CPU) の3方式でスループットと分布精度を定量比較する。

Usage:
    python -m toyomacro.voigtfit.benchmarks.poisson_benchmark
    python voigtfit/benchmarks/poisson_benchmark.py
"""

import sys
import time

import numpy as np
from scipy import stats as sp_stats

try:
    import mlx.core as mx
except ImportError:
    print("ERROR: MLX is required. Install with: pip install mlx")
    sys.exit(1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- ベンチマークパラメータ ---
LAMBDAS = [1, 5, 10, 25, 50, 100, 500, 1000]
N_THROUGHPUT = 1_000_000
N_DISTRIBUTION = 10_000_000
N_REPEATS = 10
N_WARMUP = 2


# === 3方式の実装 ===

def poisson_gaussian(lam: mx.array, key: mx.array) -> mx.array:
    """Method 1: Box-Muller Gaussian近似 (MLX GPU)"""
    z = mx.random.normal(lam.shape, key=key)
    return mx.maximum(lam + mx.sqrt(mx.maximum(lam, 1.0)) * z, 0.0)


def poisson_gaussian_preclamp(lam: mx.array, key: mx.array) -> mx.array:
    """Method 1 (pre-clamp版): clamp前の値を返す"""
    z = mx.random.normal(lam.shape, key=key)
    return lam + mx.sqrt(mx.maximum(lam, 1.0)) * z


def poisson_cornish_fisher(lam: mx.array, key: mx.array) -> mx.array:
    """Method 2: Cornish-Fisher補正 (MLX GPU)"""
    z = mx.random.normal(lam.shape, key=key)
    gamma1 = 1.0 / mx.sqrt(mx.maximum(lam, 1.0))
    z_corrected = z + gamma1 * (z * z - 1.0) / 6.0
    return mx.maximum(lam + mx.sqrt(mx.maximum(lam, 1.0)) * z_corrected, 0.0)


def poisson_cornish_fisher_preclamp(lam: mx.array, key: mx.array) -> mx.array:
    """Method 2 (pre-clamp版): clamp前の値を返す"""
    z = mx.random.normal(lam.shape, key=key)
    gamma1 = 1.0 / mx.sqrt(mx.maximum(lam, 1.0))
    z_corrected = z + gamma1 * (z * z - 1.0) / 6.0
    return lam + mx.sqrt(mx.maximum(lam, 1.0)) * z_corrected


def poisson_exact(lam: np.ndarray) -> np.ndarray:
    """Method 3: NumPy厳密Poisson (CPU, ground truth)"""
    return np.random.poisson(lam)


# === スループット測定 ===

def measure_throughput() -> dict:
    """3方式のスループットを測定 (M samples/sec)"""
    results = {}

    for lam_val in LAMBDAS:
        results[lam_val] = {}

        # --- Method 1: Gaussian ---
        lam_mx = mx.full((N_THROUGHPUT,), lam_val, dtype=mx.float32)
        # Warmup
        for i in range(N_WARMUP):
            key = mx.random.key(i)
            out = poisson_gaussian(lam_mx, key)
            mx.eval(out)

        times = []
        for i in range(N_REPEATS):
            key = mx.random.key(100 + i)
            t0 = time.perf_counter()
            out = poisson_gaussian(lam_mx, key)
            mx.eval(out)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        results[lam_val]["Gaussian"] = N_THROUGHPUT / np.median(times) / 1e6

        # --- Method 2: Cornish-Fisher ---
        for i in range(N_WARMUP):
            key = mx.random.key(i)
            out = poisson_cornish_fisher(lam_mx, key)
            mx.eval(out)

        times = []
        for i in range(N_REPEATS):
            key = mx.random.key(200 + i)
            t0 = time.perf_counter()
            out = poisson_cornish_fisher(lam_mx, key)
            mx.eval(out)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        results[lam_val]["Cornish-Fisher"] = N_THROUGHPUT / np.median(times) / 1e6

        # --- Method 3: Exact (NumPy CPU) ---
        lam_np = np.full(N_THROUGHPUT, lam_val, dtype=np.float64)
        for _ in range(N_WARMUP):
            _ = poisson_exact(lam_np)

        times = []
        for i in range(N_REPEATS):
            np.random.seed(300 + i)
            t0 = time.perf_counter()
            out = poisson_exact(lam_np)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        results[lam_val]["Exact"] = N_THROUGHPUT / np.median(times) / 1e6

    return results


# === 分布精度測定 ===

def measure_distribution() -> dict:
    """3方式の分布統計量を測定"""
    results = {}
    np.random.seed(42)

    for lam_val in LAMBDAS:
        results[lam_val] = {}

        # --- Method 1: Gaussian ---
        key = mx.random.key(42)
        lam_mx = mx.full((N_DISTRIBUTION,), lam_val, dtype=mx.float32)

        # post-clamp
        samples_post = poisson_gaussian(lam_mx, key)
        mx.eval(samples_post)
        arr_post = np.array(samples_post, dtype=np.float64)

        # pre-clamp (P(X<0) 測定用)
        key2 = mx.random.key(42)
        samples_pre = poisson_gaussian_preclamp(lam_mx, key2)
        mx.eval(samples_pre)
        arr_pre = np.array(samples_pre, dtype=np.float64)

        results[lam_val]["Gaussian"] = {
            "mean": float(np.mean(arr_post)),
            "var": float(np.var(arr_post)),
            "skewness": float(sp_stats.skew(arr_post)),
            "kurtosis": float(sp_stats.kurtosis(arr_post, fisher=True)),
            "p_neg": float(np.mean(arr_pre < 0)),
            "samples_post": arr_post,
        }

        # --- Method 2: Cornish-Fisher ---
        key3 = mx.random.key(43)
        samples_post = poisson_cornish_fisher(lam_mx, key3)
        mx.eval(samples_post)
        arr_post = np.array(samples_post, dtype=np.float64)

        key4 = mx.random.key(43)
        samples_pre = poisson_cornish_fisher_preclamp(lam_mx, key4)
        mx.eval(samples_pre)
        arr_pre = np.array(samples_pre, dtype=np.float64)

        results[lam_val]["Cornish-Fisher"] = {
            "mean": float(np.mean(arr_post)),
            "var": float(np.var(arr_post)),
            "skewness": float(sp_stats.skew(arr_post)),
            "kurtosis": float(sp_stats.kurtosis(arr_post, fisher=True)),
            "p_neg": float(np.mean(arr_pre < 0)),
            "samples_post": arr_post,
        }

        # --- Method 3: Exact ---
        lam_np = np.full(N_DISTRIBUTION, lam_val, dtype=np.float64)
        samples = poisson_exact(lam_np).astype(np.float64)

        results[lam_val]["Exact"] = {
            "mean": float(np.mean(samples)),
            "var": float(np.var(samples)),
            "skewness": float(sp_stats.skew(samples)),
            "kurtosis": float(sp_stats.kurtosis(samples, fisher=True)),
            "p_neg": 0.0,
            "samples_post": samples,
        }

    return results


# === コンソール出力 ===

def print_throughput_table(tp: dict):
    """スループットテーブル"""
    print("\n" + "=" * 70)
    print("THROUGHPUT (M samples/sec)")
    print("=" * 70)
    print(f"{'Lambda':>8}  {'Gaussian':>12}  {'Cornish-Fisher':>16}  {'Exact(CPU)':>12}")
    print("-" * 56)
    for lam in LAMBDAS:
        g = tp[lam]["Gaussian"]
        c = tp[lam]["Cornish-Fisher"]
        e = tp[lam]["Exact"]
        print(f"{lam:>8d}  {g:>12.1f}  {c:>16.1f}  {e:>12.1f}")


def print_distribution_table(dist: dict):
    """分布精度テーブル"""
    methods = ["Gaussian", "Cornish-Fisher", "Exact"]

    print("\n" + "=" * 90)
    print("DISTRIBUTION ACCURACY")
    print("=" * 90)

    # Mean
    print("\n--- Mean (expected: lambda) ---")
    print(f"{'Lambda':>8}", end="")
    for m in methods:
        print(f"  {m:>16}", end="")
    print()
    for lam in LAMBDAS:
        print(f"{lam:>8d}", end="")
        for m in methods:
            v = dist[lam][m]["mean"]
            print(f"  {v:>16.4f}", end="")
        print()

    # Variance
    print("\n--- Variance (expected: lambda) ---")
    print(f"{'Lambda':>8}", end="")
    for m in methods:
        print(f"  {m:>16}", end="")
    print()
    for lam in LAMBDAS:
        print(f"{lam:>8d}", end="")
        for m in methods:
            v = dist[lam][m]["var"]
            print(f"  {v:>16.4f}", end="")
        print()

    # Skewness
    print("\n--- Skewness (expected: 1/sqrt(lambda)) ---")
    print(f"{'Lambda':>8}  {'Theory':>8}", end="")
    for m in methods:
        print(f"  {m:>16}", end="")
    print()
    for lam in LAMBDAS:
        theory = 1.0 / np.sqrt(lam)
        print(f"{lam:>8d}  {theory:>8.4f}", end="")
        for m in methods:
            v = dist[lam][m]["skewness"]
            print(f"  {v:>16.4f}", end="")
        print()

    # Excess Kurtosis
    print("\n--- Excess Kurtosis (expected: 1/lambda) ---")
    print(f"{'Lambda':>8}  {'Theory':>8}", end="")
    for m in methods:
        print(f"  {m:>16}", end="")
    print()
    for lam in LAMBDAS:
        theory = 1.0 / lam
        print(f"{lam:>8d}  {theory:>8.4f}", end="")
        for m in methods:
            v = dist[lam][m]["kurtosis"]
            print(f"  {v:>16.4f}", end="")
        print()

    # P(X<0)
    print("\n--- P(X < 0) pre-clamp (Gaussian methods only) ---")
    print(f"{'Lambda':>8}  {'Gaussian':>12}  {'Cornish-Fisher':>16}")
    for lam in LAMBDAS:
        g = dist[lam]["Gaussian"]["p_neg"]
        c = dist[lam]["Cornish-Fisher"]["p_neg"]
        print(f"{lam:>8d}  {g:>12.6f}  {c:>16.6f}")


# === プロット生成 ===

def generate_plot(tp: dict, dist: dict, output_path: str):
    """4パネルプロット"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"Gaussian": "#2196F3", "Cornish-Fisher": "#FF9800", "Exact": "#4CAF50"}
    markers = {"Gaussian": "o", "Cornish-Fisher": "s", "Exact": "^"}

    # --- Top-left: Throughput vs lambda ---
    ax = axes[0, 0]
    for method in ["Gaussian", "Cornish-Fisher", "Exact"]:
        vals = [tp[lam][method] for lam in LAMBDAS]
        ax.plot(LAMBDAS, vals, marker=markers[method], color=colors[method],
                label=method, linewidth=2, markersize=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Throughput (M samples/sec)")
    ax.set_title("Throughput vs Lambda")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Top-right: Skewness error vs lambda ---
    ax = axes[0, 1]
    for method in ["Gaussian", "Cornish-Fisher", "Exact"]:
        deltas = []
        for lam in LAMBDAS:
            theory = 1.0 / np.sqrt(lam)
            measured = dist[lam][method]["skewness"]
            deltas.append(measured - theory)
        ax.plot(LAMBDAS, deltas, marker=markers[method], color=colors[method],
                label=method, linewidth=2, markersize=6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Skewness Error (measured - theory)")
    ax.set_title("Skewness Error vs Lambda")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: Histogram at lambda=5 ---
    ax = axes[1, 0]
    lam_hist = 5
    bins = np.arange(0, 25) - 0.5  # 整数中心のビン
    bin_centers = np.arange(0, 24)

    # 理論PMF
    pmf = sp_stats.poisson.pmf(bin_centers, lam_hist)
    ax.plot(bin_centers, pmf, "k-", linewidth=2, label="Theory (PMF)", zorder=5)

    for method in ["Gaussian", "Cornish-Fisher", "Exact"]:
        samples = dist[lam_hist][method]["samples_post"]
        ax.hist(samples, bins=bins, density=True, alpha=0.35,
                color=colors[method], label=method, edgecolor="none")

    ax.set_xlabel("X")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Distribution at lambda={lam_hist}")
    ax.legend(fontsize=8)
    ax.set_xlim(-1, 20)
    ax.grid(True, alpha=0.3)

    # --- Bottom-right: Histogram at lambda=50 ---
    ax = axes[1, 1]
    lam_hist = 50
    bins = np.arange(20, 81) - 0.5
    bin_centers = np.arange(20, 80)

    pmf = sp_stats.poisson.pmf(bin_centers, lam_hist)
    ax.plot(bin_centers, pmf, "k-", linewidth=2, label="Theory (PMF)", zorder=5)

    for method in ["Gaussian", "Cornish-Fisher", "Exact"]:
        samples = dist[lam_hist][method]["samples_post"]
        ax.hist(samples, bins=bins, density=True, alpha=0.35,
                color=colors[method], label=method, edgecolor="none")

    ax.set_xlabel("X")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Distribution at lambda={lam_hist}")
    ax.legend(fontsize=8)
    ax.set_xlim(20, 80)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {output_path}")


# === メイン ===

def main():
    import os

    # 出力先
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(script_dir), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "poisson_benchmark_results.png")

    print("=" * 70)
    print("Poisson Noise Generation Benchmark")
    print("  Methods: Gaussian (MLX), Cornish-Fisher (MLX), Exact (NumPy)")
    print(f"  Lambdas: {LAMBDAS}")
    print(f"  Throughput: {N_THROUGHPUT:,} samples x {N_REPEATS} repeats")
    print(f"  Distribution: {N_DISTRIBUTION:,} samples")
    print("=" * 70)

    # スループット測定
    print("\n[1/3] Measuring throughput...")
    tp = measure_throughput()
    print_throughput_table(tp)

    # 分布精度測定
    print("\n[2/3] Measuring distribution accuracy...")
    dist = measure_distribution()
    print_distribution_table(dist)

    # プロット生成
    print("\n[3/3] Generating plot...")
    generate_plot(tp, dist, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
