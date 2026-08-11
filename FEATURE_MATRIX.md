# Brief for feature research (paste-ready)

You are advising a systematic trader on **daily OHLC(V) features** for a next-session **long entry filter** on liquid US stocks. Suggest **new feature ideas** that fill gaps vs the live matrix below. Prefer standard definitions I can implement and hand-check.

**Do not redesign the whole system.** Propose features only. Flag data beyond single-name OHLC(V).

---

## Trader

- America/Chicago. Default **flat**. Long only when model **buy accuracy** beats noise.
- Horizon (3:00 pm CST): before → “Will **today’s** close beat prior session?” After → “Will **next** session beat latest close?”
- Payoff: win if \(\mathrm{Close}_{\text{next}} > \mathrm{Close}_{\text{as-of}}\) (hold into next close; no label for stops).
- Cares about **precision on buy days** more than recall; also **net upside after false-buy losses**.
- ML = entry filter, not portfolio construction.

---

## Success metric (feature is only useful if it helps this)

There must exist entry threshold τ on the model score such that:

1. Buy-day precision \(TP/(TP+FP) > 50\%\)
2. Precision **> base rate of UP days** in the sample
3. At least ~10 buys at that thr (not n=2 flukes)

Secondary: maximize **net gain fraction**
\(\sum_{\text{buy days}} r / \sum_{\text{all UP days}} r\), \(r = C_{\text{next}}/C_{\text{as-of}} - 1\) (losing buys pull this down).

Train discipline: **walk-forward** (train on past only). Features must be known at close of day \(t\) (**no look-ahead**).

Models (context): XGBoost classifies next-UP; may invert ranks if anti-correlated. Lasso predicts next log return (optional filter).

---

## Live feature matrix X (already built — don’t re-suggest clones)

| Name | Idea | Notes |
|------|------|--------|
| `rsi_14` | Stretch / momentum oscillator | Wilder RSI 14 |
| `obv_chg_z_20` | Volume participation | OBV change, 20-day z-score |
| `bb_dist_lower` | Distance to lower vol band | \((C - BB_{\text{lower}})/C\), BB window 14, k=1.5 |
| `vol_ratio_7_21` | Vol expansion | \(\sqrt{\text{mean GK}_7}/\sqrt{\text{mean GK}_{21}}\) (Garman–Klass from OHLC) |
| `near_high_200` | Nearness to 200-session peak | \(C_t / \max(H_{t-199\ldots t})\) (George–Hwang; ~1 = at high) |

**Already tried and dropped:** 1-day log return, raw GK vol level, EMA 5–20 spread (≈ collinear with RSI), upper BB distance (≈ RSI), CLV (same-session close-in-range; no next-UP OOS lift), `near_low_252` (52w floor proximity; ≈ inverse-RSI collinear, no uni bounce lift OOS on CRM).

**Axes covered:** stretch · volume flow · short/long vol ratio · proximity to local peak.

**Row:** one trading session of one stock. **y:** next close up/down + next log return.

---

## Constraints for candidates

- Causal at session close \(t\)
- Prefer scale-free: ratios, z-scores, % of price
- Document warm-up length (trading sessions)
- Prefer **OHLC(V) only**; if you need SPY/options/calendar, say so
- One clear economic story; state likely collinearity with the live features
- Daily bars only for now (no tick features)

---

## Gaps worth exploring

Overnight gap · liquidity (e.g. range/volume) · alt range-vol estimators · multi-day structure **if** not RSI-collinear · recovery *speed* toward high/low (Δ near_high / Δ near_low) · ATR distance above 52w low · market residual (needs index) · calendar/events (needs calendar)

---

## What to output

Rank **8–12** candidates. For each:

1. **Name** (snake_case)
2. **Idea** (one line)
3. **Definition at day t** (precise formula)
4. **Warm-up** (sessions)
5. **Data** (OHLC(V) only or other)
6. **Gap filled** vs live matrix
7. **Collinearity risk** (with which live feature?)
8. **Why it might help next-UP** (hypothesis)
9. **Why it might fail** OOS
10. **Priority** (P1/P2/P3)

Then: **top 3 to implement first** and why (discrimination / incremental / ease).

Avoid vague names (“momentum”, “sentiment”) without formulas.
