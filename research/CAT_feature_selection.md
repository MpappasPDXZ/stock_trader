# CAT Feature Selection Research

**Ticker:** CAT
**Date:** 2026-08-17
**Method:** Swap search (drop 1 baseline feature, add 1 from pool; keep if ΔAUC-PR ≥ +0.015)
**Metric:** AUC-PR (average_precision_score), oriented

---

## Summary

| Stage | Features | AUC-PR | Δ vs baseline |
|-------|----------|--------|---------------|
| Baseline 6f | rsi_14, obv_chg_z_20, bb_dist_lower, vol_ratio_7_21, near_high_200, med_term_21wk | 0.586 | — |
| **Best swap 6f** | **rsi_14, obv_chg_z_20, bb_dist_lower, vol_ratio_7_21, med_term_21wk, adx_14** | **0.702** | **+0.116** |

**Result:** Replacing `near_high_200` with `adx_14` yields AUC-PR 0.702 — a massive +0.116 lift over the 6f baseline. `adx_14` (Average Directional Index, 14-period) dominates every swap it participates in. `near_high_200` was the weakest baseline feature; removing it while adding `adx_14` is the clear winner.

---

## Swap Search (73 swaps tested)

### Top 10 swaps

| Drop | Add | AUC-PR | Δ | Decision |
|------|-----|--------|---|----------|
| near_high_200 | adx_14 | 0.702 | +0.116 | **ACCEPT** |
| vol_ratio_7_21 | adx_14 | 0.662 | +0.076 | accept |
| obv_chg_z_20 | adx_14 | 0.661 | +0.075 | accept |
| med_term_21wk | adx_14 | 0.647 | +0.061 | accept |
| near_high_200 | close_loc_value | 0.637 | +0.051 | accept |
| bb_dist_lower | adx_14 | 0.637 | +0.051 | accept |
| near_high_200 | body_ratio_ema3 | 0.633 | +0.047 | accept |
| obv_chg_z_20 | close_loc_value | 0.631 | +0.045 | accept |
| near_high_200 | fred_marshallian_k | 0.625 | +0.039 | accept |
| med_term_21wk | close_loc_value | 0.625 | +0.038 | accept |

### Key Observations

1. **`adx_14` is the strongest addition.** It appears in 4 of the top 6 swaps. Adding it (in place of any baseline feature) always improves AUC-PR.

2. **`near_high_200` is the weakest baseline feature.** Dropping it for any of `adx_14`, `close_loc_value`, `body_ratio_ema3`, or `fred_marshallian_k` all yield positive Δ.

3. **Baseline `obv_chg_z_20` and `bb_dist_lower` are indispensable.** No swap that drops either (even to add `adx_14`) beats the winning `near_high_200 → adx_14` swap.

---

## Final Model

| | Value |
|---|---|
| **Features** | rsi_14, obv_chg_z_20, bb_dist_lower, vol_ratio_7_21, med_term_21wk, adx_14 |
| **AUC-PR** | 0.702 |
| **vs baseline 6f** | +0.116 |
| **Replaced** | near_high_200 → adx_14 |

### Next Steps

- Run forward selection from this 6f set to see if adding a 7th feature (e.g., `close_loc_value`, `fred_marshallian_k`) pushes AUC-PR further.
- Run backward elimination on the 6f winner to test whether `vol_ratio_7_21` or `med_term_21wk` can be dropped.
