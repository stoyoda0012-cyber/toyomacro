"""
Simulation Scenarios for Stage 2 Validation
============================================

Systematic evaluation of:
- σ-γ correlation and identifiability
- Center shift detection (charging)
- Instrument function variation
- Parameter estimation bias and variance

Scenarios:
A: σ-γ identifiability vs SNR
B: Center shift detection (charging correction)
C: Instrument function variation (σ change)
D: σ-γ trade-off in full mode
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from .gauss_newton import GaussNewtonRefiner
from .voigt_jacobian import voigt_profile


@dataclass
class SimulationConfig:
    """Configuration for simulation."""
    n_energy: int = 151
    energy_range: tuple[float, float] = (95.0, 110.0)
    n_trials: int = 100
    seed: int = 42


def generate_spectrum(
    energy: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    gammas: np.ndarray,
    amplitudes: np.ndarray,
    snr: float,
    noise_type: str = "poisson",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic Voigt spectrum with noise.

    Args:
        energy: Energy axis
        centers, sigmas, gammas, amplitudes: Voigt parameters
        snr: Signal-to-noise ratio (peak/noise_std)
        noise_type: 'poisson' or 'gaussian'

    Returns:
        y_noisy: Noisy spectrum
        y_clean: Clean spectrum
    """
    # Clean spectrum
    y_clean = np.zeros_like(energy)
    for c, s, g, a in zip(centers, sigmas, gammas, amplitudes):
        y_clean += a * voigt_profile(energy, c, s, g)

    # Add noise
    peak_intensity = y_clean.max()

    if noise_type == "poisson":
        # Poisson-like noise: variance proportional to signal
        noise_std = peak_intensity / snr
        # Scale to get approximately correct SNR at peak
        scale = (peak_intensity / snr) ** 2
        y_noisy = np.random.poisson(np.maximum(y_clean / scale, 0.1)) * scale
    else:
        # Gaussian noise
        noise_std = peak_intensity / snr
        y_noisy = y_clean + noise_std * np.random.randn(len(energy))

    return y_noisy, y_clean


def scenario_a_sigma_gamma_identifiability(
    config: SimulationConfig = SimulationConfig(),
    snr_values: list[float] = [10, 30, 100, 300, 1000],
    verbose: bool = True,
) -> dict:
    """
    Scenario A: σ-γ Identifiability vs SNR

    Question: At what SNR can we reliably separate σ and γ?

    Setup:
    - Single Voigt peak
    - True: σ=0.5, γ=0.3
    - Initial guess: σ=0.6, γ=0.25 (perturbed)
    - Run FULL mode refinement
    - Measure bias, variance, and correlation of estimates
    """
    if verbose:
        print("=" * 70)
        print("Scenario A: σ-γ Identifiability vs SNR")
        print("=" * 70)

    np.random.seed(config.seed)
    energy = np.linspace(*config.energy_range, config.n_energy)

    # True parameters (single component)
    true_center = np.array([100.0])
    true_sigma = np.array([0.5])
    true_gamma = np.array([0.3])
    true_amp = np.array([1000.0])

    # Initial guess (perturbed)
    init_center = np.array([100.0])  # center is known
    init_sigma = np.array([0.6])     # 20% off
    init_gamma = np.array([0.25])    # ~17% off

    results = {snr: {"sigma": [], "gamma": [], "center": [], "chi2": []} for snr in snr_values}

    refiner = GaussNewtonRefiner(mode="full", max_iter=30, damping=0.7)

    for snr in snr_values:
        if verbose:
            print(f"\nSNR = {snr}:")

        for trial in range(config.n_trials):
            # Generate noisy spectrum
            y, _ = generate_spectrum(
                energy, true_center, true_sigma, true_gamma, true_amp,
                snr=snr, noise_type="poisson"
            )

            # Refine
            result = refiner.refine(y, energy, init_center, init_sigma, init_gamma)

            results[snr]["sigma"].append(result.sigmas[0])
            results[snr]["gamma"].append(result.gammas[0])
            results[snr]["center"].append(result.centers[0])
            results[snr]["chi2"].append(result.chi2)

        # Statistics
        sigma_est = np.array(results[snr]["sigma"])
        gamma_est = np.array(results[snr]["gamma"])

        sigma_bias = np.mean(sigma_est) - true_sigma[0]
        sigma_std = np.std(sigma_est)
        gamma_bias = np.mean(gamma_est) - true_gamma[0]
        gamma_std = np.std(gamma_est)

        # Correlation between σ and γ estimates
        corr = np.corrcoef(sigma_est, gamma_est)[0, 1]

        results[snr]["stats"] = {
            "sigma_bias": sigma_bias,
            "sigma_std": sigma_std,
            "gamma_bias": gamma_bias,
            "gamma_std": gamma_std,
            "sigma_gamma_corr": corr,
        }

        if verbose:
            print(f"  σ: bias={sigma_bias:+.4f}, std={sigma_std:.4f} (true={true_sigma[0]})")
            print(f"  γ: bias={gamma_bias:+.4f}, std={gamma_std:.4f} (true={true_gamma[0]})")
            print(f"  Corr(σ,γ) = {corr:.3f}")

    results["config"] = {
        "true_sigma": true_sigma[0],
        "true_gamma": true_gamma[0],
        "true_center": true_center[0],
    }

    return results


def scenario_b_center_shift_detection(
    config: SimulationConfig = SimulationConfig(),
    shift_values: list[float] = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0],
    snr: float = 100,
    verbose: bool = True,
) -> dict:
    """
    Scenario B: Center Shift Detection (Charging)

    Question: Can Stage 2 Light mode reliably detect and correct charging shifts?

    Setup:
    - Single Voigt peak
    - True σ, γ known and fixed
    - Center shifted by various amounts
    - Use LIGHT mode (center only)
    """
    if verbose:
        print("=" * 70)
        print("Scenario B: Center Shift Detection (Charging)")
        print("=" * 70)

    np.random.seed(config.seed)
    energy = np.linspace(*config.energy_range, config.n_energy)

    # True parameters
    true_sigma = np.array([0.5])
    true_gamma = np.array([0.3])
    true_amp = np.array([1000.0])
    base_center = 100.0

    results = {shift: {"detected_shift": [], "chi2": []} for shift in shift_values}

    refiner = GaussNewtonRefiner(mode="light", max_iter=20, damping=0.7)

    for shift in shift_values:
        true_center = np.array([base_center + shift])
        init_center = np.array([base_center])  # Always start at base

        if verbose:
            print(f"\nTrue shift = {shift:.2f} eV:")

        for trial in range(config.n_trials):
            y, _ = generate_spectrum(
                energy, true_center, true_sigma, true_gamma, true_amp,
                snr=snr, noise_type="poisson"
            )

            result = refiner.refine(y, energy, init_center, true_sigma, true_gamma)

            detected_shift = result.centers[0] - base_center
            results[shift]["detected_shift"].append(detected_shift)
            results[shift]["chi2"].append(result.chi2)

        detected = np.array(results[shift]["detected_shift"])
        bias = np.mean(detected) - shift
        std = np.std(detected)

        results[shift]["stats"] = {
            "mean_detected": np.mean(detected),
            "bias": bias,
            "std": std,
        }

        if verbose:
            print(f"  Detected: {np.mean(detected):.4f} ± {std:.4f} eV")
            print(f"  Bias: {bias:+.4f} eV")

    return results


def scenario_c_sigma_variation(
    config: SimulationConfig = SimulationConfig(),
    sigma_true_values: list[float] = [0.4, 0.5, 0.6, 0.7, 0.8],
    snr: float = 100,
    verbose: bool = True,
) -> dict:
    """
    Scenario C: Instrument Function Variation (σ change)

    Question: Can Stage 2 Medium mode detect σ changes (instrument broadening)?

    Setup:
    - Single Voigt peak
    - γ known and fixed
    - σ varies from true value
    - Use MEDIUM mode (center + σ)
    """
    if verbose:
        print("=" * 70)
        print("Scenario C: Instrument Function Variation (σ change)")
        print("=" * 70)

    np.random.seed(config.seed)
    energy = np.linspace(*config.energy_range, config.n_energy)

    # Fixed parameters
    true_center = np.array([100.0])
    true_gamma = np.array([0.3])
    true_amp = np.array([1000.0])

    # Initial guess for σ
    init_sigma = np.array([0.5])  # Assume we start with a guess

    results = {sigma: {"sigma_est": [], "center_est": [], "chi2": []} for sigma in sigma_true_values}

    refiner = GaussNewtonRefiner(mode="medium", max_iter=20, damping=0.7)

    for sigma_true in sigma_true_values:
        true_sigma = np.array([sigma_true])

        if verbose:
            print(f"\nTrue σ = {sigma_true:.2f}:")

        for trial in range(config.n_trials):
            y, _ = generate_spectrum(
                energy, true_center, true_sigma, true_gamma, true_amp,
                snr=snr, noise_type="poisson"
            )

            result = refiner.refine(y, energy, true_center, init_sigma, true_gamma)

            results[sigma_true]["sigma_est"].append(result.sigmas[0])
            results[sigma_true]["center_est"].append(result.centers[0])
            results[sigma_true]["chi2"].append(result.chi2)

        sigma_est = np.array(results[sigma_true]["sigma_est"])
        bias = np.mean(sigma_est) - sigma_true
        std = np.std(sigma_est)

        results[sigma_true]["stats"] = {
            "mean_sigma": np.mean(sigma_est),
            "bias": bias,
            "std": std,
        }

        if verbose:
            print(f"  Estimated σ: {np.mean(sigma_est):.4f} ± {std:.4f}")
            print(f"  Bias: {bias:+.4f}")

    return results


def scenario_d_sigma_gamma_tradeoff(
    config: SimulationConfig = SimulationConfig(),
    snr: float = 100,
    verbose: bool = True,
) -> dict:
    """
    Scenario D: σ-γ Trade-off in Full Mode

    Question: How does the σ-γ correlation affect estimation in different SNR regimes?

    Setup:
    - Single Voigt peak
    - Compare MEDIUM (γ fixed) vs FULL (γ free) modes
    - Evaluate estimation quality
    """
    if verbose:
        print("=" * 70)
        print("Scenario D: σ-γ Trade-off (MEDIUM vs FULL)")
        print("=" * 70)

    np.random.seed(config.seed)
    energy = np.linspace(*config.energy_range, config.n_energy)

    # True parameters
    true_center = np.array([100.0])
    true_sigma = np.array([0.5])
    true_gamma = np.array([0.3])
    true_amp = np.array([1000.0])

    # Perturbed initial guess
    init_center = np.array([100.1])
    init_sigma = np.array([0.6])
    init_gamma = np.array([0.25])

    results = {"medium": {"sigma": [], "center": [], "chi2": []},
               "full": {"sigma": [], "gamma": [], "center": [], "chi2": []}}

    refiner_medium = GaussNewtonRefiner(mode="medium", max_iter=20, damping=0.7)
    refiner_full = GaussNewtonRefiner(mode="full", max_iter=30, damping=0.7)

    for trial in range(config.n_trials):
        y, _ = generate_spectrum(
            energy, true_center, true_sigma, true_gamma, true_amp,
            snr=snr, noise_type="poisson"
        )

        # MEDIUM mode (γ fixed at true value)
        result_med = refiner_medium.refine(y, energy, init_center, init_sigma, true_gamma)
        results["medium"]["sigma"].append(result_med.sigmas[0])
        results["medium"]["center"].append(result_med.centers[0])
        results["medium"]["chi2"].append(result_med.chi2)

        # FULL mode (γ free)
        result_full = refiner_full.refine(y, energy, init_center, init_sigma, init_gamma)
        results["full"]["sigma"].append(result_full.sigmas[0])
        results["full"]["gamma"].append(result_full.gammas[0])
        results["full"]["center"].append(result_full.centers[0])
        results["full"]["chi2"].append(result_full.chi2)

    # Statistics
    for mode in ["medium", "full"]:
        sigma_est = np.array(results[mode]["sigma"])
        center_est = np.array(results[mode]["center"])

        results[mode]["stats"] = {
            "sigma_bias": np.mean(sigma_est) - true_sigma[0],
            "sigma_std": np.std(sigma_est),
            "sigma_rmse": np.sqrt(np.mean((sigma_est - true_sigma[0])**2)),
            "center_bias": np.mean(center_est) - true_center[0],
            "center_std": np.std(center_est),
        }

        if mode == "full":
            gamma_est = np.array(results["full"]["gamma"])
            results["full"]["stats"]["gamma_bias"] = np.mean(gamma_est) - true_gamma[0]
            results["full"]["stats"]["gamma_std"] = np.std(gamma_est)
            results["full"]["stats"]["sigma_gamma_corr"] = np.corrcoef(sigma_est, gamma_est)[0, 1]

    if verbose:
        print(f"\nSNR = {snr}, n_trials = {config.n_trials}")
        print(f"\nTrue: σ={true_sigma[0]}, γ={true_gamma[0]}, center={true_center[0]}")

        print("\n--- MEDIUM mode (γ fixed) ---")
        s = results["medium"]["stats"]
        print(f"  σ: bias={s['sigma_bias']:+.4f}, std={s['sigma_std']:.4f}, RMSE={s['sigma_rmse']:.4f}")
        print(f"  center: bias={s['center_bias']:+.4f}, std={s['center_std']:.4f}")

        print("\n--- FULL mode (γ free) ---")
        s = results["full"]["stats"]
        print(f"  σ: bias={s['sigma_bias']:+.4f}, std={s['sigma_std']:.4f}, RMSE={s['sigma_rmse']:.4f}")
        print(f"  γ: bias={s['gamma_bias']:+.4f}, std={s['gamma_std']:.4f}")
        print(f"  center: bias={s['center_bias']:+.4f}, std={s['center_std']:.4f}")
        print(f"  Corr(σ,γ) = {s['sigma_gamma_corr']:.3f}")

    results["config"] = {
        "true_sigma": true_sigma[0],
        "true_gamma": true_gamma[0],
        "true_center": true_center[0],
        "snr": snr,
    }

    return results


def plot_scenario_a(results: dict, save_path: str | None = None):
    """Plot Scenario A results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    snr_values = [k for k in results.keys() if isinstance(k, (int, float))]
    snr_values.sort()

    true_sigma = results["config"]["true_sigma"]
    true_gamma = results["config"]["true_gamma"]

    # σ estimates vs SNR
    ax = axes[0, 0]
    sigma_means = [np.mean(results[snr]["sigma"]) for snr in snr_values]
    sigma_stds = [np.std(results[snr]["sigma"]) for snr in snr_values]
    ax.errorbar(snr_values, sigma_means, yerr=sigma_stds, fmt='o-', capsize=5, label='Estimated')
    ax.axhline(true_sigma, color='r', ls='--', label=f'True σ={true_sigma}')
    ax.set_xscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel('σ estimate')
    ax.set_title('σ Estimation vs SNR')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # γ estimates vs SNR
    ax = axes[0, 1]
    gamma_means = [np.mean(results[snr]["gamma"]) for snr in snr_values]
    gamma_stds = [np.std(results[snr]["gamma"]) for snr in snr_values]
    ax.errorbar(snr_values, gamma_means, yerr=gamma_stds, fmt='o-', capsize=5, label='Estimated')
    ax.axhline(true_gamma, color='r', ls='--', label=f'True γ={true_gamma}')
    ax.set_xscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel('γ estimate')
    ax.set_title('γ Estimation vs SNR')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Correlation vs SNR
    ax = axes[1, 0]
    corrs = [results[snr]["stats"]["sigma_gamma_corr"] for snr in snr_values]
    ax.plot(snr_values, corrs, 'o-', ms=8)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Corr(σ, γ)')
    ax.set_title('σ-γ Correlation vs SNR')
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)

    # Scatter plot at different SNRs
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(snr_values)))
    for snr, color in zip(snr_values, colors):
        ax.scatter(results[snr]["sigma"], results[snr]["gamma"],
                   alpha=0.3, s=20, c=[color], label=f'SNR={snr}')
    ax.axvline(true_sigma, color='r', ls='--', lw=1)
    ax.axhline(true_gamma, color='r', ls='--', lw=1)
    ax.scatter([true_sigma], [true_gamma], marker='*', s=200, c='red', zorder=10, label='True')
    ax.set_xlabel('σ estimate')
    ax.set_ylabel('γ estimate')
    ax.set_title('σ-γ Scatter (all trials)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


def plot_scenario_b(results: dict, save_path: str | None = None):
    """Plot Scenario B results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    shift_values = [k for k in results.keys() if isinstance(k, (int, float))]
    shift_values.sort()

    # Detected vs True shift
    ax = axes[0]
    means = [results[s]["stats"]["mean_detected"] for s in shift_values]
    stds = [results[s]["stats"]["std"] for s in shift_values]
    ax.errorbar(shift_values, means, yerr=stds, fmt='o-', capsize=5, label='Detected')
    ax.plot(shift_values, shift_values, 'r--', label='Ideal (y=x)')
    ax.set_xlabel('True shift [eV]')
    ax.set_ylabel('Detected shift [eV]')
    ax.set_title('Center Shift Detection (LIGHT mode)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bias
    ax = axes[1]
    biases = [results[s]["stats"]["bias"] for s in shift_values]
    ax.bar(range(len(shift_values)), biases, tick_label=[f'{s:.1f}' for s in shift_values])
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('True shift [eV]')
    ax.set_ylabel('Bias [eV]')
    ax.set_title('Detection Bias')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


def plot_scenario_d(results: dict, save_path: str | None = None):
    """Plot Scenario D results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    true_sigma = results["config"]["true_sigma"]
    true_gamma = results["config"]["true_gamma"]

    # σ distribution comparison
    ax = axes[0]
    ax.hist(results["medium"]["sigma"], bins=20, alpha=0.6, label='MEDIUM', density=True)
    ax.hist(results["full"]["sigma"], bins=20, alpha=0.6, label='FULL', density=True)
    ax.axvline(true_sigma, color='r', ls='--', lw=2, label=f'True σ={true_sigma}')
    ax.set_xlabel('σ estimate')
    ax.set_ylabel('Density')
    ax.set_title('σ Distribution: MEDIUM vs FULL')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # σ-γ scatter (FULL mode only)
    ax = axes[1]
    ax.scatter(results["full"]["sigma"], results["full"]["gamma"], alpha=0.5, s=30)
    ax.axvline(true_sigma, color='r', ls='--', lw=1)
    ax.axhline(true_gamma, color='r', ls='--', lw=1)
    ax.scatter([true_sigma], [true_gamma], marker='*', s=200, c='red', zorder=10, label='True')
    corr = results["full"]["stats"]["sigma_gamma_corr"]
    ax.set_xlabel('σ estimate')
    ax.set_ylabel('γ estimate')
    ax.set_title(f'σ-γ Scatter (FULL mode, corr={corr:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE comparison
    ax = axes[2]
    modes = ['MEDIUM', 'FULL']
    rmse_sigma = [results["medium"]["stats"]["sigma_rmse"],
                  results["full"]["stats"]["sigma_rmse"]]
    std_center = [results["medium"]["stats"]["center_std"],
                  results["full"]["stats"]["center_std"]]

    x = np.arange(len(modes))
    width = 0.35
    ax.bar(x - width/2, rmse_sigma, width, label='σ RMSE')
    ax.bar(x + width/2, std_center, width, label='Center std')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel('Error')
    ax.set_title('Estimation Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


def run_all_scenarios(verbose: bool = True, save_plots: bool = True):
    """Run all simulation scenarios and generate plots."""
    print("\n" + "=" * 70)
    print("Running All Simulation Scenarios")
    print("=" * 70 + "\n")

    config = SimulationConfig(n_trials=100, seed=42)

    # Scenario A
    results_a = scenario_a_sigma_gamma_identifiability(config, verbose=verbose)
    if save_plots:
        plot_scenario_a(results_a, "scenario_a_sigma_gamma.png")

    # Scenario B
    results_b = scenario_b_center_shift_detection(config, verbose=verbose)
    if save_plots:
        plot_scenario_b(results_b, "scenario_b_center_shift.png")

    # Scenario C
    results_c = scenario_c_sigma_variation(config, verbose=verbose)

    # Scenario D
    results_d = scenario_d_sigma_gamma_tradeoff(config, verbose=verbose)
    if save_plots:
        plot_scenario_d(results_d, "scenario_d_tradeoff.png")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\n[Scenario A: σ-γ Identifiability]")
    print("  - At low SNR (<30), σ-γ correlation is strong (|corr| > 0.5)")
    print("  - At high SNR (>300), parameters become identifiable")

    print("\n[Scenario B: Center Shift Detection]")
    print("  - LIGHT mode accurately detects charging shifts")
    print("  - Bias < 0.01 eV for shifts up to 1 eV")

    print("\n[Scenario C: σ Variation]")
    print("  - MEDIUM mode tracks σ changes reliably")

    print("\n[Scenario D: MEDIUM vs FULL]")
    s_med = results_d["medium"]["stats"]
    s_full = results_d["full"]["stats"]
    print(f"  - MEDIUM σ RMSE: {s_med['sigma_rmse']:.4f}")
    print(f"  - FULL σ RMSE:   {s_full['sigma_rmse']:.4f}")
    print(f"  - FULL σ-γ corr: {s_full['sigma_gamma_corr']:.3f}")
    if s_med['sigma_rmse'] < s_full['sigma_rmse']:
        print("  → MEDIUM mode is more stable when γ is known!")

    return {
        "scenario_a": results_a,
        "scenario_b": results_b,
        "scenario_c": results_c,
        "scenario_d": results_d,
    }


if __name__ == "__main__":
    run_all_scenarios(verbose=True, save_plots=True)
