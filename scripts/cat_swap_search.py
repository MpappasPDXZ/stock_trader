"""Single-feature swap search for CAT.

For each of the 6 baseline features, try replacing it with each of the 12 extra features.
Keep a swap if AUC-PR improves by > 0.015. Only one swap is kept (best improvement).
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_ml.calibration import calibrate
from market_ml.models import list_models

TICKER = "CAT"
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


def calibrate_and_get(features, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  features ({len(features)}): {features}")
    print(f"{'='*60}")

    name = label.replace(" ", "_").replace("→", "->").replace("+", "p")
    result = calibrate(TICKER, features=features, name=name)
    if not result.get("ok"):
        print(f"  FAILED: {result.get('error')}")
        return None

    model_id = result["model_id"]
    models = list_models(TICKER)
    m = next((x for x in models if x["model_id"] == model_id), None)
    if m is None:
        return None

    auc = m.get("oriented_auc")
    print(f"  AUC-PR: {auc:.4f}")
    return {"auc": auc, "model_id": model_id, "features": features}


# Baseline
print("\n\n###### BASELINE 6f ######")
baseline = calibrate_and_get(BASELINE_6F, "baseline 6f")
if baseline is None:
    print("FATAL: baseline failed")
    sys.exit(1)

baseline_auc = baseline["auc"]
print(f"\nBaseline AUC-PR: {baseline_auc:.4f}")

# Try all swaps
print("\n\n###### SWAP SEARCH ######")
results = []
for drop_feat in BASELINE_6F:
    for add_feat in EXTRA_FEATURES:
        trial = [f for f in BASELINE_6F if f != drop_feat] + [add_feat]
        label = f"swap {drop_feat} → {add_feat}"
        r = calibrate_and_get(trial, label)
        if r is None:
            continue
        delta = r["auc"] - baseline_auc
        results.append({
            "drop": drop_feat,
            "add": add_feat,
            "features": trial,
            "auc": r["auc"],
            "delta": delta,
        })
        print(f"  Δ={delta:+.4f}")

# Sort by delta descending
results.sort(key=lambda x: x["delta"], reverse=True)

print("\n\n###### RESULTS (top 10) ######")
for r in results[:10]:
    tag = "✓ KEEP" if r["delta"] > THRESHOLD else "  toss"
    print(f"  {tag}  {r['drop']:20s} → {r['add']:20s}  AUC-PR={r['auc']:.4f}  Δ={r['delta']:+.4f}")

# Best swap
best = results[0] if results else None
if best and best["delta"] > THRESHOLD:
    print(f"\n\n###### ACCEPTED SWAP ######")
    print(f"  Drop: {best['drop']}")
    print(f"  Add:  {best['add']}")
    print(f"  AUC-PR: {best['auc']:.4f} (baseline {baseline_auc:.4f}, Δ={best['delta']:+.4f})")
    print(f"  Features: {best['features']}")

    # Save the winning model
    print("\nSaving final model...")
    result = calibrate(TICKER, features=best["features"], name=f"CAT_swap_{best['drop']}_for_{best['add']}")
    print(f"  model_id: {result.get('model_id')}")
else:
    print(f"\n\n###### NO SWAP MEETS THRESHOLD ######")
    print(f"  Best: {best['drop']} → {best['add']}  Δ={best['delta']:+.4f} (need > {THRESHOLD})")
    print(f"  Keeping baseline 6f")

# Write results
out = {
    "ticker": TICKER,
    "baseline_auc": baseline_auc,
    "threshold": THRESHOLD,
    "all_swaps": results,
    "best": best,
    "accepted": best is not None and best["delta"] > THRESHOLD,
}
out_path = os.path.join(os.path.dirname(__file__), "cat_swap_search_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved to: {out_path}")
