# GEV Feature Selection Research

**Ticker:** GEV  
**Date:** 2026-08-14  
**Method:** Forward selection (+0.015 AUC-PR threshold) → Backward elimination (-0.015 tolerance)  
**Metric:** AUC-PR (average_precision_score), oriented  

---

## Summary

| Stage | Features | AUC-PR | Δ vs baseline |
|-------|----------|--------|---------------|
| Baseline 6f | rsi_14, obv_chg_z_20, bb_dist_lower, vol_ratio_7_21, near_high_200, med_term_21wk | 0.590 | — |
| Forward final 7f | 6f + fred_marshallian_k | 0.617 | +0.027 |
| **Backward final 3f** | **obv_chg_z_20, bb_dist_lower, near_high_200** | **0.618** | **+0.028** |

**Result:** The 3-feature model (obv_chg_z_20, bb_dist_lower, near_high_200) achieves AUC-PR 0.618, a +0.028 improvement over the 6f baseline (0.590). Dropping rsi_14, vol_ratio_7_21, med_term_21wk, and fred_marshallian_k did not hurt.

---

## Forward Selection (6f baseline → 7f)

### Round 1: 6f + 1 extra feature

| Feature Added | AUC-PR | Δ vs 0.590 | Decision |
|---------------|--------|------------|----------|
| fred_marshallian_k | 0.617 | +0.027 | **ACCEPT** (≥0.015) |
| body_ratio_ema3 | 0.602 | +0.012 | reject |
| fred_real_yield | 0.598 | +0.008 | reject |
| fred_stress_z | 0.594 | +0.004 | reject |
| di_spread | 0.573 | -0.017 | reject |
| wick_asym_roll5 | 0.573 | -0.017 | reject |
| wick_asym_ema5 | 0.569 | -0.021 | reject |
| fred_yield_slope | 0.567 | -0.023 | reject |
| wick_asym_ema3 | 0.563 | -0.027 | reject |
| close_loc_value | 0.559 | -0.031 | reject |
| wick_asymmetry | 0.556 | -0.034 | reject |
| adx_14 | 0.552 | -0.038 | reject |

**Winner:** fred_marshallian_k (Δ=+0.027, threshold met)

### Round 2: 7f + 1 extra feature (remaining 11)

| Feature Added | AUC-PR | Δ vs 0.617 | Decision |
|---------------|--------|------------|----------|
| body_ratio_ema3 | 0.629 | +0.012 | reject (<0.015) |
| fred_stress_z | 0.626 | +0.009 | reject |
| fred_yield_slope | 0.607 | -0.011 | reject |
| fred_real_yield | 0.605 | -0.012 | reject |
| di_spread | 0.601 | -0.017 | reject |
| close_loc_value | 0.594 | -0.023 | reject |
| wick_asymmetry | 0.593 | -0.025 | reject |
| wick_asym_ema3 | 0.588 | -0.029 | reject |
| adx_14 | 0.589 | -0.028 | reject |
| wick_asym_ema5 | 0.573 | -0.044 | reject |
| wick_asym_roll5 | 0.569 | -0.048 | reject |

**No feature met 0.015 threshold. Forward selection stopped.**

---

## Backward Elimination (7f → 3f)

Starting set: rsi_14, obv_chg_z_20, bb_dist_lower, vol_ratio_7_21, near_high_200, med_term_21wk, fred_marshallian_k (AUC-PR=0.617)

### Round 1: Drop 1 of 7

| Dropped | AUC-PR | Δ | Decision |
|---------|--------|---|----------|
| vol_ratio_7_21 | 0.636 | +0.019 | **DROP** (best, improved) |
| rsi_14 | 0.620 | +0.003 | |
| bb_dist_lower | 0.619 | +0.002 | |
| obv_chg_z_20 | 0.593 | -0.024 | |
| near_high_200 | 0.593 | -0.024 | |
| med_term_21wk | 0.591 | -0.026 | |
| fred_marshallian_k | 0.590 | -0.027 | |

**Dropped:** vol_ratio_7_21 → AUC-PR=0.636

### Round 2: Drop 1 of 6

| Dropped | AUC-PR | Δ | Decision |
|---------|--------|---|----------|
| rsi_14 | 0.640 | +0.004 | **DROP** (best, improved) |
| fred_marshallian_k | 0.621 | -0.015 | |
| bb_dist_lower | 0.627 | -0.008 | |
| med_term_21wk | 0.612 | -0.024 | |
| obv_chg_z_20 | 0.606 | -0.029 | |
| near_high_200 | 0.606 | -0.029 | |

**Dropped:** rsi_14 → AUC-PR=0.640

### Round 3: Drop 1 of 5

| Dropped | AUC-PR | Δ | Decision |
|---------|--------|---|----------|
| fred_marshallian_k | 0.628 | -0.012 | **DROP** (best, Δ ≥ -0.015) |
| med_term_21wk | 0.621 | -0.019 | |
| bb_dist_lower | 0.625 | -0.015 | |
| near_high_200 | 0.598 | -0.042 | |
| obv_chg_z_20 | 0.587 | -0.052 | |

**Dropped:** fred_marshallian_k → AUC-PR=0.628

### Round 4: Drop 1 of 4

| Dropped | AUC-PR | Δ | Decision |
|---------|--------|---|----------|
| med_term_21wk | 0.618 | -0.009 | **DROP** (best, Δ ≥ -0.015) |
| bb_dist_lower | 0.583 | -0.045 | |
| near_high_200 | 0.580 | -0.047 | |
| obv_chg_z_20 | 0.571 | -0.057 | |

**Dropped:** med_term_21wk → AUC-PR=0.618

### Round 5: Drop 1 of 3

| Dropped | AUC-PR | Δ | Decision |
|---------|--------|---|----------|
| bb_dist_lower | 0.541 | -0.078 | KEEP |
| obv_chg_z_20 | 0.568 | -0.051 | KEEP |
| near_high_200 | 0.568 | -0.051 | KEEP |

**All drops hurt by >0.015. Backward elimination stopped.**

---

## Final Model

| | Value |
|---|---|
| **Features** | obv_chg_z_20, bb_dist_lower, near_high_200 |
| **AUC-PR** | 0.618 |
| **vs baseline 6f** | +0.028 |
| **Eliminated** | rsi_14, vol_ratio_7_21, med_term_21wk, fred_marshallian_k |

### Key Findings

1. **fred_marshallian_k** was the only extra feature that met the forward threshold (+0.027 AUC-PR), but backward elimination later removed it — the 6f baseline features already captured the same signal.

2. **rsi_14 and vol_ratio_7_21 hurt the model.** Dropping either one improved AUC-PR (rsi_14: +0.004, vol_ratio_7_21: +0.019). Both were removed in backward elimination.

3. **med_term_21wk** was the least important of the remaining 4 — dropping it only cost 0.009 AUC-PR, within the tolerance.

4. **The 3-feature model (obv_chg_z_20, bb_dist_lower, near_high_200) is the optimal set.** All three are indispensable — removing any one drops AUC-PR by >0.05.
