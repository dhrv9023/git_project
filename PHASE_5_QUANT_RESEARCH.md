# Phase 5 — Quantitative Research Platform
## A Research-Paper-Style Engineering Document

**Project:** StockBuddy Atelier  
**Phase:** 5 — Quantitative Research Platform  
**Date:** 2026-08-08  
**Author:** Senior Quant Research Engineer

---

## Abstract

This document details the implementation of 17 institutional-grade quantitative metrics added to StockBuddy in Phase 5. Each metric is explained from first principles with its mathematical formula, intuition, Python implementation, visualization strategy, interpretation thresholds, and rationale for use by professional quants. The metrics are computed in `ml/quant_analytics.py` and served via `POST /api/v5/quant`.

---

## 1. Information Coefficient (IC)

**Formula:**
```
IC_t = SpearmanRankCorr(signal_{t-W:t}, forward_return_{t-W:t})
```
where W = 60-day rolling window.

**Intuition:** IC measures how well a signal (regime allocation weight) predicts future returns. A perfect predictor has IC = 1.0; a random predictor has IC = 0. Unlike Pearson correlation, Spearman rank correlation is robust to outliers — critical for fat-tailed financial return distributions.

**Python Implementation:**
```python
from scipy.stats import spearmanr
rho, pval = spearmanr(signal_window, forward_return_window)
```

**Visualization:** Rolling time-series line chart with zero-line reference and colored fill (green above, red below).

**Interpretation:**
- IC > 0.05 → good signal
- IC > 0.10 → excellent signal
- IC-IR (IC mean / IC std) > 0.5 → consistent alpha signal

**Why Quants Use It:** IC is the primary metric at Renaissance Technologies, AQR, and Two Sigma for evaluating factor predictiveness. A high IC with low IC-IR suggests the signal works but inconsistently — requiring position sizing adjustments.

---

## 2. Sharpe Ratio

**Formula:**
```
Sharpe = (μ_r - R_f) / σ_r × √252
```
where μ_r = mean daily strategy return, R_f = daily risk-free rate (5% annual ÷ 252), σ_r = std of daily returns.

**Intuition:** Sharpe normalizes return by total volatility. It answers: "How much excess return do I get per unit of risk?"

**Python Implementation:**
```python
rf_daily = 0.05 / 252
excess   = strategy_returns - rf_daily
sharpe   = excess.mean() / excess.std() * np.sqrt(252)
```

**Visualization:** KPI card + rolling 60-day Sharpe time-series line chart.

**Interpretation:**
- Sharpe > 1.0 → good
- Sharpe > 2.0 → excellent
- Sharpe > 3.0 → institutional-grade (rare in real markets)

**Why Quants Use It:** The Sharpe ratio is the universal currency of quantitative finance. Every fund allocator, prime broker, and risk committee uses it to compare strategies on a risk-adjusted basis.

---

## 3. Sortino Ratio

**Formula:**
```
Sortino = (μ_r - R_f) / σ_downside × √252
```
where σ_downside = std of negative returns only.

**Intuition:** Unlike Sharpe, Sortino penalizes only *downside* volatility. A strategy that spikes upward frequently will have a higher Sortino than Sharpe. This is a fairer measure for asymmetric (trend-following) strategies.

**Python Implementation:**
```python
downside_returns = returns[returns < 0]
sortino = excess.mean() / downside_returns.std() * np.sqrt(252)
```

**Visualization:** KPI card alongside Sharpe for direct comparison.

**Interpretation:**
- Sortino > Sharpe → strategy has positive skew (good)
- Sortino > 1.5 → strong asymmetric return profile

**Why Quants Use It:** Endowments and pension funds often prefer Sortino because they care more about losses than volatility per se. Bridgewater Associates uses downside-adjusted metrics in all-weather portfolio design.

---

## 4. Calmar Ratio

**Formula:**
```
Calmar = CAGR% / |Maximum Drawdown%|
```

**Intuition:** Calmar divides the annualised growth rate by the worst peak-to-trough loss. It captures the relationship between long-term compounding and worst-case pain. Higher Calmar = better return per unit of drawdown risk.

**Python Implementation:**
```python
calmar = cagr_pct / abs(max_drawdown_pct)
```

**Visualization:** KPI card.

**Interpretation:**
- Calmar > 0.5 → acceptable
- Calmar > 1.0 → good
- Calmar > 3.0 → excellent (top CTAs)

**Why Quants Use It:** Commodity Trading Advisors (CTAs) and managed futures funds report Calmar as a primary performance metric because their risk budget is defined in terms of acceptable drawdown.

---

## 5. Maximum Drawdown (MDD)

**Formula:**
```
MDD = min[(V_t - peak_t) / peak_t] × 100%
peak_t = max(V_0, V_1, ..., V_t)
```

**Intuition:** MDD is the largest observed peak-to-trough decline in portfolio value. It represents the worst historical loss an investor would have experienced.

**Python Implementation:**
```python
peak = np.maximum.accumulate(equity)
drawdown = (equity - peak) / peak * 100
mdd = drawdown.min()
```

**Visualization:** Filled area chart ("underwater equity") — the area below zero represents periods of loss relative to the prior peak.

**Interpretation:**
- MDD < -10% → manageable
- MDD < -20% → moderate
- MDD < -50% → severe (most retail investors capitulate here)

**Why Quants Use It:** Risk managers use MDD to define stop-loss rules and position sizing via the Kelly criterion. A strategy's MDD also determines its maximum leverage before ruin probability becomes unacceptable.

---

## 6. CAGR (Compound Annual Growth Rate)

**Formula:**
```
CAGR = (V_T / V_0)^(252 / T) - 1
```
where T = number of trading days, V_T = terminal equity, V_0 = initial capital.

**Intuition:** CAGR is the geometric mean annual return — the rate at which an investment would have grown if it compounded smoothly every year. It accounts for compounding unlike simple arithmetic return.

**Python Implementation:**
```python
cagr = (equity[-1] / equity[0]) ** (252 / n_days) - 1
```

**Visualization:** KPI card with strategy vs Buy & Hold comparison.

**Interpretation:**
- CAGR > S&P 500 (~10-12% annually) → market-beating
- CAGR > 20% → exceptional (top quartile hedge funds)

**Why Quants Use It:** CAGR is used as the numerator of the Calmar ratio and as the primary return metric for long-only and multi-asset strategies where time horizon matters.

---

## 7. Alpha (Jensen's Alpha)

**Formula:**
```
α = (μ_strat - R_f) - β × (μ_market - R_f)   [annualised]
```

**Intuition:** Alpha measures excess return above what could be explained by the market (beta). A strategy with alpha > 0 generates returns that cannot be attributed to simply holding the market index — evidence of genuine skill or edge.

**Python Implementation:**
```python
from sklearn.linear_model import LinearRegression
reg  = LinearRegression().fit(market_returns.reshape(-1,1), strategy_returns)
beta = reg.coef_[0]
alpha = (strategy_mean - rf) - beta * (market_mean - rf)
alpha_annual = alpha * 252 * 100  # in percent
```

**Visualization:** KPI card color-coded by sign.

**Interpretation:**
- Alpha > 0 → generates value above market beta exposure
- Alpha > 5% annually → institutionally significant
- Alpha < 0 → market exposure alone explains or exceeds returns

**Why Quants Use It:** Jensen's Alpha from the Capital Asset Pricing Model (CAPM) is the baseline test used by allocators to evaluate whether a manager justifies active fees over an index fund.

---

## 8. Beta

**Formula:**
```
β = Cov(r_strategy, r_market) / Var(r_market)
```

**Intuition:** Beta measures a strategy's systematic market sensitivity. Beta = 1.0 means the strategy moves in lockstep with SPY. Beta = 0.5 means half the market sensitivity. Beta < 0 means the strategy is negatively correlated with markets.

**Python Implementation:**
```python
cov = np.cov(strategy_returns, market_returns)[0, 1]
beta = cov / np.var(market_returns)
```

**Visualization:** KPI card.

**Interpretation:**
- β ≈ 0 → market-neutral strategy
- β ≈ 1 → fully correlated with market
- β < 0 → hedge / tail risk strategy

**Why Quants Use It:** Beta decomposition tells investors how much risk they are buying vs. what the market offers for free. High-beta strategies are not worth active management fees unless alpha is also high.

---

## 9. Rolling Volatility (30-day Annualised)

**Formula:**
```
σ_t(30) = std(r_{t-29}, ..., r_t) × √252 × 100%
```

**Intuition:** Rolling volatility reveals how risk changes over time — it spikes during crises (COVID-19 March 2020, GFC 2008) and compresses during calm markets. A regime-aware strategy should automatically reduce exposure when volatility rises.

**Python Implementation:**
```python
roll_vol = daily_returns.rolling(30).std() * np.sqrt(252) * 100
```

**Visualization:** Amber-filled line chart showing temporal volatility evolution.

**Interpretation:**
- Rolling Vol < 10% → low-volatility regime
- Rolling Vol 10-25% → normal market
- Rolling Vol > 30% → crisis / stress regime

**Why Quants Use It:** Volatility targeting is a core technique at AQR Capital. Strategies like Risk Parity dynamically scale position sizes so that annualised vol remains constant (~10-15%), regardless of which asset class is leading.

---

## 10. Rolling Sharpe (60-day)

**Formula:**
```
RollingSharpe_t = Sharpe(r_{t-59}, ..., r_t)
```

**Intuition:** A single Sharpe ratio over the full backtest period can hide periods of poor performance. The rolling Sharpe exposes whether strategy quality is consistent over time or concentrated in a short period.

**Python Implementation:**
```python
roll_sharpe = strategy_ret.rolling(60).apply(
    lambda x: (x.mean() - rf) / x.std() * np.sqrt(252)
)
```

**Visualization:** Green-filled line chart with Sharpe = 1 reference line.

**Interpretation:**
- Consistently above 1.0 → stable alpha-generating strategy
- Large negative periods → regime-dependent edge, likely trend-following

**Why Quants Use It:** Institutional allocators require monthly Sharpe reports. A strategy with declining rolling Sharpe signals factor decay — triggering research reviews at firms like Two Sigma.

---

## 11. Walk-Forward Optimisation

**Formula / Method:**
```
For fold f in {1,...,5}:
  Train on: days[0 : fold × n/6]
  Test on:  days[fold × n/6 : (fold+1) × n/6]
  Grid search: {(bull_thresh, neutral_thresh)} = {(1.0,0.5), (0.8,0.4), (0.6,0.3)}
  Select: argmax Sharpe over test fold
```

**Intuition:** Walk-forward optimisation prevents overfitting by fitting strategy parameters on past data and evaluating them on unseen future data. It simulates how a quant would actually deploy and update a strategy in real-time.

**Python Implementation:**
```python
for fold in range(n_folds):
    train_end = fold_size * (fold + 1)
    best_params = grid_search(train_data[:train_end], threshold_grid)
    test_sharpe = evaluate(test_data[train_end:], best_params)
```

**Visualization:** Table with per-fold Sharpe, period, and optimal thresholds.

**Interpretation:**
- Consistent Sharpe across folds → genuine, non-overfitted edge
- Fold Sharpe variance > 2× mean → overfitted, regime-dependent
- Mean fold Sharpe > 1.0 → institutionally deployable

**Why Quants Use It:** Walk-forward is the gold standard for strategy validation in algorithmic trading. It is the method required by hedge funds when presenting to prime brokers and institutional allocators.

---

## 12. Cross-Validation (IC Stability)

**Formula / Method:**
```
TimeSeriesSplit(n_splits=5):
  IC_fold_k = SpearmanRankCorr(signal_k, forward_return_k)
IC_mean ± IC_std across folds
```

**Intuition:** Cross-validation checks whether the IC signal is stable across different time periods. A high IC mean with low IC std indicates the signal works consistently, not just in one lucky period.

**Python Implementation:**
```python
from sklearn.model_selection import TimeSeriesSplit
tss = TimeSeriesSplit(n_splits=5)
cv_ics = [spearmanr(signal[te], fwd_ret[te])[0] for _, te in tss.split(signal)]
```

**Visualization:** Bar chart, green bars for positive IC folds, red bars for negative.

**Interpretation:**
- All folds positive IC → strongly consistent signal
- Mixed folds → regime-dependent signal
- IC std / IC mean > 2 → unstable, not deployable at scale

**Why Quants Use It:** AQR's factor research requires IC consistency tests across market regimes. A factor that only works in bull markets has dangerous regime-conditional exposure.

---

## 13. Statistical Significance (t-test on IC)

**Formula:**
```
H₀: IC = 0
t = IC_mean / (IC_std / √n)
p-value = P(T > |t|) under t-distribution
```

**Intuition:** Statistical significance testing determines whether an observed IC is likely due to chance. An IC of 0.04 over 60 observations may not be significant; over 1,000 observations it likely is.

**Python Implementation:**
```python
from scipy.stats import ttest_1samp
t_stat, p_value = ttest_1samp(ic_series, popmean=0)
significant = p_value < 0.05
```

**Visualization:** Colored badge (`SIGNIFICANT ✓` or `NOT SIG ✗`) on the IC panel.

**Interpretation:**
- p < 0.05 → reject H₀ at 95% confidence
- p < 0.01 → reject H₀ at 99% confidence
- p > 0.10 → insufficient evidence the signal is non-random

**Why Quants Use It:** Academic finance requires p < 0.05 for publication; industry quants typically require p < 0.01 due to multiple testing concerns (Harvey, Liu, Zhu 2016 showed most factors fail this threshold).

---

## 14. Regime Confidence

**Formula:**
```
conf_i = 1 - d_i(centroid_i) / max(distances)
where d_i = distance from sample i to its assigned K-Means centroid
```

**Intuition:** A data point in the dense core of a cluster has high confidence (low distance to centroid). A point near the boundary between clusters has low confidence — the regime classification is ambiguous and transitions are likely.

**Python Implementation:**
```python
distances = kmeans.transform(scaled_features)  # shape (N, k)
min_dist  = distances.min(axis=1)
confidence = 1 - min_dist / distances.max()
```

**Visualization:** Horizontal bar chart with regime-colored bars for each regime's average confidence.

**Interpretation:**
- Confidence > 0.8 → strong regime identification
- Confidence < 0.5 → regime boundary, increased transition probability
- Low confidence on current day → reduce position sizing

**Why Quants Use It:** Regime confidence is used in probabilistic portfolio construction. A low-confidence regime day might trigger a hedge or reduce leverage, while high-confidence bull regime days justify full equity exposure.

---

## 15. Monte Carlo Simulation

**Formula:**
```
Path_k: S_t = S_0 × ∏_{i=1}^{T} (1 + ε_i)
where ε_i ~ N(μ_strategy, σ_strategy)
Percentiles: P5, P25, P50, P75, P95 across 1,000 paths
```

**Intuition:** Monte Carlo simulation generates thousands of possible future equity paths by sampling from the historical return distribution. It provides a probabilistic distribution of outcomes — far more honest than a single backtest line.

**Python Implementation:**
```python
np.random.seed(42)
paths = np.cumprod(
    1 + np.random.normal(mu, sigma, (n_paths, n_days)), axis=1
) * initial_capital
p5, p50, p95 = np.percentile(paths, [5, 50, 95], axis=0)
```

**Visualization:** Fan chart with 5 percentile bands (P5/P25/P50/P75/P95) showing the cone of uncertainty.

**Interpretation:**
- P50 > initial capital → strategy has positive expected value
- P5 > 0 → even worst-case scenarios avoid total ruin
- Wide P5-P95 spread → high outcome uncertainty, consider diversification
- Prob(Profit) = % of paths ending above initial capital

**Why Quants Use It:** Monte Carlo is required for VaR (Value at Risk) and CVaR (Conditional VaR) calculations under Basel III banking regulations. Portfolio managers at BlackRock use MC simulations for long-term asset allocation stress testing.

---

## 16. Feature Importance (Permutation Importance)

**Formula:**
```
Importance_f = (Inertia_shuffled_f - Inertia_baseline) / Inertia_baseline
where Inertia = sum of squared distances to assigned cluster centroids
```

**Intuition:** Permutation importance measures how much cluster quality degrades when a feature's values are randomly shuffled (destroying its predictive relationship). A feature that matters a lot will cause a large inertia increase when shuffled.

**Python Implementation:**
```python
baseline_inertia = compute_inertia(kmeans, features)
for feature_idx, feature_name in enumerate(feature_cols):
    shuffled = features.copy()
    shuffled[:, feature_idx] = np.random.permutation(shuffled[:, feature_idx])
    importance = (compute_inertia(kmeans, shuffled) - baseline_inertia) / baseline_inertia
```

**Visualization:** Horizontal bar chart with gradient fill, features sorted by importance.

**Interpretation:**
- High importance → feature drives regime differentiation (e.g., Vol20 in crisis regimes)
- Low importance → feature is redundant, could be dropped
- All features similar → regime structure is well-distributed

**Why Quants Use It:** Factor attribution models at Goldman Sachs Quantitative Research use permutation importance to decide which risk factors to include in multi-factor alpha models. Irrelevant factors add noise without improving prediction.

---

## 17. Regime Transition Matrix (Markov Chain)

**Formula:**
```
P[i → j] = count(regime_t = i AND regime_{t+1} = j) / count(regime_t = i)
P is a 6×6 row-stochastic matrix where ∑_j P[i,j] = 1
```

**Intuition:** The transition matrix reveals the persistence and evolution of market regimes. A high diagonal value P[i,i] means a regime is persistent (markets stay in it for a while). Off-diagonal elements show which regimes tend to follow each other.

**Python Implementation:**
```python
trans = np.zeros((6, 6))
for t in range(len(labels) - 1):
    trans[labels[t], labels[t+1]] += 1
prob = trans / trans.sum(axis=1, keepdims=True)
```

**Visualization:** 6×6 heatmap table with cyan intensity proportional to transition probability.

**Interpretation:**
- High P[Bull → Bull] → bull trends are persistent (buy and hold works)
- High P[Bear → Bear] → bear markets are persistent (cut losses quickly)
- High P[Sideways → Bear] → sideways regimes often precede bear markets (early warning)
- Low diagonal → high regime instability, trend-following strategies underperform

**Why Quants Use It:** Regime-switching models (Hamilton 1989) are widely used at JP Morgan, Barclays, and state pension funds for dynamic asset allocation. The transition matrix provides a forward-looking probability distribution over next-period regime state, enabling Bayesian updating of tactical allocations.

---

## Summary Table

| # | Metric | Formula Summary | Visualization | Interpretation Threshold |
|---|---|---|---|---|
| 1 | IC | Spearman(signal, fwd_ret) | Rolling line | > 0.05 = good |
| 2 | Sharpe | μ_excess/σ × √252 | KPI + rolling | > 1.0 = good |
| 3 | Sortino | μ_excess/σ_down × √252 | KPI | > Sharpe = positive skew |
| 4 | Calmar | CAGR/|MDD| | KPI | > 1.0 = good |
| 5 | MDD | min(drawdown series) | Underwater area | < -20% = severe |
| 6 | CAGR | (V_T/V_0)^(252/T) - 1 | KPI | > 10% = market-beating |
| 7 | Alpha | μ_excess - β×μ_mkt | KPI | > 0 = genuine skill |
| 8 | Beta | Cov(r,r_mkt)/Var(r_mkt) | KPI | ~0 = market-neutral |
| 9 | Rolling Vol | std(30d) × √252 | Line chart | > 30% = crisis |
| 10 | Rolling Sharpe | Sharpe(60d window) | Line chart | > 1.0 consistently |
| 11 | Walk-Forward | Grid search per fold | Table | Low fold variance |
| 12 | Cross-Val | IC per TimeSeriesSplit | Bar chart | All folds positive |
| 13 | Stat Sig | t-test on IC (H₀: IC=0) | Badge | p < 0.05 |
| 14 | Regime Conf | 1 - dist/max_dist | Bar per regime | > 0.8 = strong |
| 15 | Monte Carlo | 1000 paths, N(μ,σ) | Fan chart | P50 > 10k |
| 16 | Feature Imp | Permutation vs inertia | H-bar chart | Ranked by magnitude |
| 17 | Transition Mx | Markov P[i→j] | 6×6 heatmap | High diagonal = persistent |

---

## API Reference

```
POST /api/v5/quant
Content-Type: application/json

{
  "ticker":     "AAPL",
  "start_date": "2020-01-01",
  "end_date":   "2024-12-31"
}

Response keys:
  performance        → dict: sharpe, sortino, calmar, mdd, cagr, alpha, beta
  ic                 → dict: mean, std, ir, series, dates, t_stat, p_value, significant
  rolling            → dict: dates, volatility, sharpe, equity_strat, drawdown
  monte_carlo        → dict: p5/p25/p50/p75/p95 series, prob_profit
  transition_matrix  → dict: labels, matrix (6×6)
  feature_importance → list: [{feature, importance}] sorted desc
  regime_confidence  → dict: current, by_regime
  walk_forward       → dict: folds, sharpe_mean, sharpe_std
  cross_validation   → dict: fold_ics, ic_mean, ic_std
```

## Implementation Files

| File | Purpose |
|---|---|
| `ml/quant_analytics.py` | All 17 metric computations (self-contained) |
| `app.py` | `POST /api/v5/quant` Flask route |
| `index.html` | Quantitative Research Lab UI section |
| `PHASE_5_QUANT_RESEARCH.md` | This document |
