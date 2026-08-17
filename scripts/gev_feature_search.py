"""Forward selection + backward elimination for GEV feature set.

Threshold: ΔAUC-PR ≥ 0.015 to keep a forward addition.
Backward elimination: drop if AUC-PR doesn't drop by more than 0.015.
"""

import json
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_ml.calibration import calibrate

TICKER = "GEV"
THRESHOLD = 0.015

BASELINE_6F = [
    "rsi_14", "obv_chg_z_20", "bb_dist_lower",
    "vol_ratio_7_21", "near_high_200", "med_term_21wk",
]

EXTRA_FEATURES = [
    "adx_14", "di_spread", "body_ratio_ema3",
    "wick_asymmetry", "wick_asym_ema3", "wick_asym_ema5",
    "wick_asym_roll5", "close_loc_value",
    "fred_yield_slope", "fred_stress_z", "fred_real_yield",
    "fred_marshallian_k",
]

results = {"forward": [], "backward": []}


def calibrate_and_get_auc(features, label):
    """Calibrate with given features, return (auc, model_id, discrim, precision, n_wins, n_buys)."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  features ({len(features)}): {features}")
    print(f"{'='*60}")

    name = label.replace(" ", "_").replace("→", "->").replace("+", "p")
    result = calibrate(TICKER, features=features, name=name)
    if not result.get("ok"):
        print(f"  FAILED: {result.get('error')}")
        return None, None, None, None, None, None

    model_id = result["model_id"]

    # Fetch calibration status to get AUC-PR
    from market_ml.models import list_models
    models = list_models(TICKER)
    m = next((x for x in models if x["model_id"] == model_id), None)
    if m is None:
        print(f"  model not found in list: {model_id}")
        return None, None, None, None, None, None

    auc = m.get("oriented_auc")
    discrim = m.get("is_discriminating")
    precision = m.get("precision")
    n_wins = m.get("n_wins")
    n_buys = m.get("n_buys")

    print(f"  AUC-PR: {auc:.4f}  discrim={discrim}  prec={precision}  {n_wins}/{n_buys}")
    return auc, model_id, discrim, precision, n_wins, n_buys


# ── Step 1: Baseline 6f ──
print("\n\n###### STEP 1: BASELINE 6f ######")
baseline_auc, baseline_mid, baseline_disc, baseline_prec, baseline_w, baseline_b = \
    calibrate_and_get_auc(BASELINE_6F, "baseline 6f")
results["baseline"] = {
    "features": BASELINE_6F,
    "auc": baseline_auc,
    "model_id": baseline_mid,
    "discrim": baseline_disc,
    "precision": baseline_prec,
    "n_wins": baseline_w,
    "n_buys": baseline_b,
}

if baseline_auc is None:
    print("FATAL: baseline calibration failed")
    sys.exit(1)

current_features = list(BASELINE_6F)
current_auc = baseline_auc
remaining = list(EXTRA_FEATURES)

# ── Step 2: Forward selection ──
print("\n\n###### STEP 2: FORWARD SELECTION ######")
round_num = 0
while remaining:
    round_num += 1
    print(f"\n--- Forward round {round_num} (current: {len(current_features)}f, AUC-PR={current_auc:.4f}) ---")
    round_results = []

    for feat in remaining:
        trial_features = current_features + [feat]
        auc, mid, disc, prec, nw, nb = calibrate_and_get_auc(
            trial_features,
            f"forward round {round_num}: +{feat}"
        )
        if auc is None:
            continue
        delta = auc - current_auc
        round_results.append({
            "feature": feat,
            "auc": auc,
            "delta": delta,
            "model_id": mid,
            "discrim": disc,
            "precision": prec,
            "n_wins": nw,
            "n_buys": nb,
        })
        results["forward"].append({
            "round": round_num,
            "feature": feat,
            "features": trial_features,
            "auc": auc,
            "delta": delta,
        })

    if not round_results:
        break

    # Sort by delta descending
    round_results.sort(key=lambda x: x["delta"], reverse=True)
    best = round_results[0]
    print(f"\n  Best this round: +{best['feature']}  Δ={best['delta']:+.4f}  AUC-PR={best['auc']:.4f}")

    if best["delta"] >= THRESHOLD:
        print(f"  ✓ ACCEPT (Δ ≥ {THRESHOLD})")
        current_features = current_features + [best["feature"]]
        current_auc = best["auc"]
        remaining.remove(best["feature"])
        results["forward_best"] = {
            "round": round_num,
            "feature": best["feature"],
            "features": list(current_features),
            "auc": current_auc,
        }
    else:
        print(f"  ✗ REJECT (Δ < {THRESHOLD}), stopping forward selection")
        break

print(f"\n=== Forward selection complete: {len(current_features)}f, AUC-PR={current_auc:.4f} ===")
print(f"  features: {current_features}")
results["forward_final"] = {
    "features": list(current_features),
    "auc": current_auc,
}

# ── Step 3: Backward elimination ──
print("\n\n###### STEP 3: BACKWARD ELIMINATION ######")
backward_features = list(current_features)
backward_auc = current_auc
eliminated = []
round_num = 0

while len(backward_features) > 1:
    round_num += 1
    print(f"\n--- Backward round {round_num} (current: {len(backward_features)}f, AUC-PR={backward_auc:.4f}) ---")
    round_results = []

    for feat in backward_features:
        trial_features = [f for f in backward_features if f != feat]
        auc, mid, disc, prec, nw, nb = calibrate_and_get_auc(
            trial_features,
            f"backward round {round_num}: -{feat}"
        )
        if auc is None:
            continue
        delta = auc - backward_auc  # negative means dropping hurts
        round_results.append({
            "feature": feat,
            "auc": auc,
            "delta": delta,
            "model_id": mid,
        })
        results["backward"].append({
            "round": round_num,
            "dropped": feat,
            "features": trial_features,
            "auc": auc,
            "delta": delta,
        })

    if not round_results:
        break

    # Sort by delta descending — highest delta means dropping helped most
    round_results.sort(key=lambda x: x["delta"], reverse=True)
    best = round_results[0]
    print(f"\n  Best to drop: -{best['feature']}  Δ={best['delta']:+.4f}  AUC-PR={best['auc']:.4f}")

    if best["delta"] >= -THRESHOLD:
        # Dropping this feature doesn't hurt by more than threshold
        print(f"  ✓ DROP (Δ ≥ -{THRESHOLD})")
        backward_features = [f for f in backward_features if f != best["feature"]]
        backward_auc = best["auc"]
        eliminated.append(best["feature"])
        results.setdefault("backward_dropped", []).append({
            "round": round_num,
            "feature": best["feature"],
            "features": list(backward_features),
            "auc": backward_auc,
        })
    else:
        print(f"  ✗ KEEP (Δ < -{THRESHOLD}), stopping backward elimination")
        break

print(f"\n=== Backward elimination complete: {len(backward_features)}f, AUC-PR={backward_auc:.4f} ===")
print(f"  features: {backward_features}")
print(f"  eliminated: {eliminated}")
results["backward_final"] = {
    "features": list(backward_features),
    "auc": backward_auc,
    "eliminated": eliminated,
}

# ── Summary ──
results["summary"] = {
    "baseline_6f_auc": baseline_auc,
    "forward_final_auc": current_auc,
    "backward_final_auc": backward_auc,
    "baseline_features": BASELINE_6F,
    "final_features": backward_features,
    "forward_added": [f for f in current_features if f not in BASELINE_6F],
    "backward_eliminated": eliminated,
}

# Write results JSON
out_path = os.path.join(os.path.dirname(__file__), "gev_feature_search_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\n###### FINAL SUMMARY ######")
print(f"  Baseline 6f AUC-PR:    {baseline_auc:.4f}")
print(f"  Forward final AUC-PR:  {current_auc:.4f}")
print(f"  Backward final AUC-PR: {backward_auc:.4f}")
print(f"  Final features ({len(backward_features)}): {backward_features}")
print(f"  Results saved to: {out_path}")
