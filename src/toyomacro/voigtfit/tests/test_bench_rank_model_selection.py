"""CI smoke coverage for the Gate 4 benchmark (frozen README §4.4-P2).

Only the smoke subset and a reduced S3 slice run in CI; the full frozen
grid is on-demand (``--mode full``) and never part of the normal suite.
"""

import time

from toyomacro.voigtfit.benchmarks.bench_rank_model_selection import (
    SMOKE_GRID,
    _cells,
    aggregate,
    run_cell,
    run_slice_so,
)


def test_smoke_grid_is_fast_and_never_overclaims():
    t0 = time.perf_counter()
    rows = [run_cell(cell) for cell in _cells(SMOKE_GRID)]
    wall = time.perf_counter() - t0

    agg = aggregate(rows)
    assert wall < 60.0, f"P2 violated: smoke took {wall:.1f}s"
    # no confident claim of MORE structure than the truth
    assert agg["no_confident_overfit_claim_rate"]["rate"] == 1.0
    # frozen S4, direct: IC/rank disagreement never yields a single-K
    # assertion (None when the subset is empty)
    s4 = agg["S4_no_single_K_assertion_under_IC_rank_disagreement"]
    assert s4["rate"] in (None, 1.0)
    # S1 on the smoke subset (well-separated, high SNR, K>1)
    s1 = agg["S1_true_k_in_supported_at_sep>=1FWHM_snr>=100"]
    assert s1["n"] >= 1 and s1["rate"] == 1.0
    # every cell produced a verdict from the frozen vocabulary
    assert sum(agg["verdict_distribution"].values()) == agg["n_cells"]


def test_s3_reduced_slice_structured_doublet_not_disfavored():
    result = run_slice_so(seeds=range(2), snrs=(100,))
    assert result["S3_structured_not_disfavored_rate"] == 1.0
