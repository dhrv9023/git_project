"""
ml/quant_analytics.py — Phase 5: Quantitative Research Analytics Engine

Computes 17 institutional-grade quantitative metrics:
  1.  Information Coefficient (IC)
  2.  Sharpe Ratio
  3.  Sortino Ratio
  4.  Calmar Ratio
  5.  Maximum Drawdown
  6.  CAGR
  7.  Alpha (Jensen's)
  8.  Beta (vs SPY)
  9.  Rolling Volatility (30-day annualised)
  10. Rolling Sharpe (60-day)
  11. Walk-Forward Optimisation (threshold grid)
  12. Cross-Validation (IC stability, TimeSeriesSplit)
  13. Statistical Significance (t-test on IC)
  14. Regime Confidence (centroid distance)
  15. Monte Carlo Simulation (1,000 paths)
  16. Feature Importance (permutation vs KMeans inertia)
  17. Regime Transition Matrix (Markov P[i->j])

Public API
----------
    compute_quant_research_report(ticker, start_date, end_date) -> dict

Design
------
Self-contained: imports only numpy, pandas, scipy, sklearn + yfinance.
Does NOT import from app.py to avoid circular dependencies.
The Flask route in app.py does a lazy `from ml.quant_analytics import ...`.
"""

import math
import warnings
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_FREE_ANNUAL = 0.05
TRADING_DAYS     = 252
MC_PATHS         = 1000        # paths kept low for synchronous safety
MC_HORIZON       = 252         # 1-year forward projection
BENCHMARK_TICKER = 'SPY'


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from yfinance; return Close-only DataFrame."""
    import yfinance as yf
    df = yf.download(
        tickers=ticker, start=start, end=end,
        auto_adjust=True, progress=False, group_by='column'
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} ({start} → {end})")
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=-1)
        except Exception:
            df.columns = ['_'.join(str(x) for x in c) for c in df.columns]
    df.columns = [c.title() for c in df.columns]
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    return df[['Close']].dropna()


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-returns, RSI-14, EMA-20, MACD histogram, Vol-20, momentum features."""
    d = df.copy()
    d['LogReturn'] = np.log(d['Close']).diff()

    # RSI-14
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI14'] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # EMA-20
    d['EMA20'] = d['Close'].ewm(span=20, adjust=False).mean()

    # MACD histogram
    ema12 = d['Close'].ewm(span=12, adjust=False).mean()
    ema26 = d['Close'].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    d['MACD_hist'] = macd - macd.ewm(span=9, adjust=False).mean()

    # Regime features
    d['Vol20']          = d['LogReturn'].rolling(20).std()
    d['Ret10']          = d['Close'].pct_change(10) * 100
    d['RSI_dev']        = d['RSI14'] - 50
    d['EMA_slope']      = (d['EMA20'] - d['EMA20'].shift(10)) / (d['EMA20'].shift(10).replace(0, 1) + 1e-9) * 100
    d['MACD_hist_norm'] = d['MACD_hist'] / (d['Close'].rolling(20).mean().replace(0, 1) * 0.01 + 1e-9)

    return d.dropna()


def _classify_regimes(df: pd.DataFrame, n: int = 6):
    """
    K-Means regime clustering on 5 standardised features.
    Returns (labels Series, feature DataFrame, fitted KMeans, fitted StandardScaler).
    Semantic ordering: cluster 0 = most bullish/calm, 5 = most bearish/stressed.
    """
    FEAT_COLS = ['RSI_dev', 'MACD_hist_norm', 'EMA_slope', 'Vol20', 'Ret10']
    feat = df[FEAT_COLS].replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    scaler = StandardScaler()
    fs = scaler.fit_transform(feat.values)
    km = KMeans(n_clusters=n, random_state=42, n_init=20)
    raw = km.fit_predict(fs)

    # Semantic ordering: score = 2*Ret10 + 1.5*RSI_dev - 3*Vol20
    c = km.cluster_centers_
    scores     = 2 * c[:, 4] + 1.5 * c[:, 0] - 3 * c[:, 3]
    sorted_c   = np.argsort(-scores)
    mapping    = {int(o): int(s) for s, o in enumerate(sorted_c)}
    labels     = pd.Series([mapping[l] for l in raw], index=feat.index, name='regime')
    return labels, feat, km, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe(v):
    """Return float rounded to 4 dp, or None if NaN/Inf."""
    if v is None:
        return None
    f = float(v)
    return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)


def _sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe: (μ_excess) / σ × √252."""
    rf    = RISK_FREE_ANNUAL / TRADING_DAYS
    exc   = np.asarray(returns) - rf
    std   = exc.std()
    return float(exc.mean() / (std + 1e-12) * np.sqrt(TRADING_DAYS))


def _sortino(returns: np.ndarray) -> float:
    """Annualised Sortino: uses only downside deviation."""
    rf       = RISK_FREE_ANNUAL / TRADING_DAYS
    exc      = np.asarray(returns) - rf
    downside = returns[returns < 0]
    std_d    = downside.std() if len(downside) > 1 else 1e-9
    return float(exc.mean() / (std_d + 1e-12) * np.sqrt(TRADING_DAYS))


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown in percent (negative)."""
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / (peak + 1e-12)
    return float(dd.min() * 100)


def _cagr(equity: np.ndarray, n_days: int) -> float:
    """Compound Annual Growth Rate in percent."""
    if n_days <= 0 or equity[0] <= 0:
        return 0.0
    return float(((equity[-1] / equity[0]) ** (TRADING_DAYS / n_days) - 1) * 100)


def _calmar(cagr_pct: float, mdd_pct: float) -> float:
    """Calmar = CAGR% / |MaxDrawdown%|."""
    return float(cagr_pct / abs(mdd_pct)) if mdd_pct != 0 else 0.0


def _regime_weights(labels: pd.Series) -> pd.Series:
    """Map regime label → tactical equity weight (Phase 1 allocation matrix)."""
    w = pd.Series(0.0, index=labels.index)
    w[labels.isin([0, 3])] = 1.0
    w[labels.isin([1, 2])] = 0.5
    w[labels.isin([4, 5])] = 0.0
    return w


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def compute_quant_research_report(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Entry point for Phase 5 quantitative research report.

    Parameters
    ----------
    ticker     : str  — e.g. 'AAPL'
    start_date : str  — e.g. '2020-01-01'
    end_date   : str  — e.g. '2024-12-31'

    Returns
    -------
    dict with keys: performance, ic, rolling, monte_carlo,
                    transition_matrix, feature_importance,
                    regime_confidence, walk_forward, cross_validation
    """
    log.info(f"[Phase5] Computing quant research report: {ticker} {start_date}→{end_date}")

    # ── 1. Data ingestion ─────────────────────────────────────────────────────
    df       = _engineer(_fetch(ticker, start_date, end_date))
    bench_df = _engineer(_fetch(BENCHMARK_TICKER, start_date, end_date))

    common   = df.index.intersection(bench_df.index)
    df       = df.loc[common]
    bench_df = bench_df.loc[common]

    # ── 2. Regime classification ──────────────────────────────────────────────
    labels, feat, km, scaler = _classify_regimes(df)
    common2   = df.index.intersection(labels.index)
    df_r      = df.loc[common2]
    labels_r  = labels.loc[common2]

    daily_ret = df_r['LogReturn'].fillna(0)
    bench_ret = bench_df['LogReturn'].loc[common2].fillna(0)

    # Strategy returns (Phase 1 allocation matrix + transaction costs)
    weights    = _regime_weights(labels_r)
    tc_cost    = weights.diff().abs().fillna(0) * 0.001
    strat_ret  = weights.shift(1).fillna(0) * daily_ret - tc_cost

    equity     = 10_000 * (1 + strat_ret).cumprod()
    bh_equity  = 10_000 * (1 + daily_ret).cumprod()
    n_days     = len(equity)

    strat_np = strat_ret.values
    bh_np    = daily_ret.values
    bench_np = bench_ret.values

    # ── 3. Scalar performance metrics ─────────────────────────────────────────
    sharpe_s  = _sharpe(strat_np)
    sortino_s = _sortino(strat_np)
    mdd_s     = _max_drawdown(equity.values)
    cagr_s    = _cagr(equity.values, n_days)
    calmar_s  = _calmar(cagr_s, mdd_s)
    sharpe_bh = _sharpe(bh_np)
    mdd_bh    = _max_drawdown(bh_equity.values)
    cagr_bh   = _cagr(bh_equity.values, n_days)

    # Alpha & Beta (Jensen's alpha vs SPY)
    reg   = LinearRegression().fit(bench_np.reshape(-1, 1), strat_np)
    beta  = float(reg.coef_[0])
    rf_d  = RISK_FREE_ANNUAL / TRADING_DAYS
    alpha = (np.mean(strat_np) - rf_d - beta * (np.mean(bench_np) - rf_d)) * TRADING_DAYS * 100

    # ── 4. Information Coefficient (rolling 60-day Spearman) ─────────────────
    fwd_ret   = daily_ret.shift(-1)
    ic_vals, ic_dates = [], []
    W = 60
    for i in range(W, len(weights) - 1):
        w_w   = weights.iloc[i - W:i].values
        f_w   = fwd_ret.iloc[i - W:i].values
        mask  = ~np.isnan(f_w)
        if mask.sum() > 10:
            rho, _ = spearmanr(w_w[mask], f_w[mask])
            ic_vals.append(0.0 if math.isnan(float(rho)) else float(rho))
            ic_dates.append(weights.index[i].strftime('%Y-%m-%d'))

    ic_mean = float(np.mean(ic_vals)) if ic_vals else 0.0
    ic_std  = float(np.std(ic_vals))  if ic_vals else 1e-9
    ic_ir   = ic_mean / (ic_std + 1e-12)

    # Statistical significance (t-test: H0: IC = 0)
    if len(ic_vals) > 2:
        t_stat, p_val = ttest_1samp(ic_vals, 0)
    else:
        t_stat, p_val = 0.0, 1.0
    ic_sig = bool(p_val < 0.05)

    # ── 5. Rolling metrics ────────────────────────────────────────────────────
    step  = max(1, n_days // 250)   # cap frontend data points
    idx_s = slice(None, None, step)

    roll_vol = daily_ret.rolling(30).std() * np.sqrt(TRADING_DAYS) * 100
    roll_sh  = strat_ret.rolling(60).apply(
        lambda x: _sharpe(x.values) if len(x) > 5 else 0.0, raw=False
    )

    # Drawdown series
    pk_s = np.maximum.accumulate(equity.values)
    dd_s = (equity.values - pk_s) / (pk_s + 1e-12) * 100

    def _ser(s): return [(_safe(v)) for v in s]

    rolling_out = {
        'dates':        [d.strftime('%Y-%m-%d') for d in daily_ret.index[idx_s]],
        'volatility':   _ser(roll_vol.iloc[idx_s]),
        'sharpe':       _ser(roll_sh.iloc[idx_s]),
        'equity_strat': [round(float(v), 2) for v in equity.iloc[idx_s]],
        'equity_bh':    [round(float(v), 2) for v in bh_equity.iloc[idx_s]],
        'equity_dates': [d.strftime('%Y-%m-%d') for d in equity.index[idx_s]],
        'drawdown':     [round(float(v), 3) for v in dd_s[idx_s]],
    }

    # ── 6. Monte Carlo (1,000 paths, 252-day horizon) ─────────────────────────
    np.random.seed(42)
    mu_mc, sig_mc = float(strat_ret.mean()), float(strat_ret.std())
    paths = 10_000 * np.cumprod(
        1 + np.random.normal(mu_mc, sig_mc, (MC_PATHS, MC_HORIZON)), axis=1
    )
    mc_step = max(1, MC_HORIZON // 80)
    mc_out  = {
        'days':  list(range(0, MC_HORIZON, mc_step)),
        'p5':    [round(float(v), 2) for v in np.percentile(paths, 5,  axis=0)[::mc_step]],
        'p25':   [round(float(v), 2) for v in np.percentile(paths, 25, axis=0)[::mc_step]],
        'p50':   [round(float(v), 2) for v in np.percentile(paths, 50, axis=0)[::mc_step]],
        'p75':   [round(float(v), 2) for v in np.percentile(paths, 75, axis=0)[::mc_step]],
        'p95':   [round(float(v), 2) for v in np.percentile(paths, 95, axis=0)[::mc_step]],
        'final_p5':     round(float(np.percentile(paths[:, -1], 5)), 2),
        'final_p50':    round(float(np.percentile(paths[:, -1], 50)), 2),
        'final_p95':    round(float(np.percentile(paths[:, -1], 95)), 2),
        'prob_profit':  round(float((paths[:, -1] > 10_000).mean() * 100), 1),
    }

    # ── 7. Regime Transition Matrix ───────────────────────────────────────────
    N = 6
    trans = np.zeros((N, N))
    arr   = labels_r.values
    for i in range(len(arr) - 1):
        rf, rt = int(arr[i]), int(arr[i + 1])
        if 0 <= rf < N and 0 <= rt < N:
            trans[rf, rt] += 1
    row_s = trans.sum(axis=1, keepdims=True)
    prob  = np.where(row_s > 0, trans / (row_s + 1e-12), 0)
    RNAMES = ['Trending Bull', 'Overbought', 'Sideways', 'Recovery', 'Bear', 'High Vol']
    tm_out = {
        'labels': RNAMES,
        'matrix': [[round(float(prob[i, j]), 3) for j in range(N)] for i in range(N)],
    }

    # ── 8. Feature Importance (permutation vs KMeans inertia) ────────────────
    FEAT_COLS = ['RSI_dev', 'MACD_hist_norm', 'EMA_slope', 'Vol20', 'Ret10']
    fv  = feat[FEAT_COLS].replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna().values
    fs  = scaler.transform(fv)
    bl  = km.predict(fs)
    base_inertia = sum(
        float(np.sum((fs[bl == k] - km.cluster_centers_[k]) ** 2))
        for k in range(km.n_clusters)
    )
    np.random.seed(42)
    fi_out = []
    for fi, fn in enumerate(FEAT_COLS):
        sh = fs.copy()
        sh[:, fi] = np.random.permutation(sh[:, fi])
        sl = km.predict(sh)
        sh_inertia = sum(
            float(np.sum((sh[sl == k] - km.cluster_centers_[k]) ** 2))
            for k in range(km.n_clusters)
        )
        fi_out.append({
            'feature':    fn,
            'importance': round((sh_inertia - base_inertia) / (base_inertia + 1e-12), 4),
        })
    fi_out.sort(key=lambda x: -x['importance'])

    # ── 9. Regime Confidence ──────────────────────────────────────────────────
    fc = feat.loc[labels_r.index, FEAT_COLS].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0).values
    dists   = km.transform(scaler.transform(fc))
    min_d   = dists.min(axis=1)
    conf    = 1 - min_d / (dists.max() + 1e-12)
    conf_by = {
        RNAMES[rid]: round(float(conf[labels_r.values == rid].mean()), 3)
        for rid in range(N)
        if (labels_r.values == rid).sum() > 0
    }
    # fill missing regimes
    for rn in RNAMES:
        conf_by.setdefault(rn, 0.0)

    # ── 10. Walk-Forward Optimisation (threshold grid) ────────────────────────
    fold_sz  = n_days // 6
    THRESH   = [(1.0, 0.5), (0.8, 0.4), (0.6, 0.3)]
    wf_folds = []
    for fold in range(5):
        ts, te = fold_sz * (fold + 1), min(fold_sz * (fold + 2), n_days)
        if te - ts < 20:
            continue
        lab_te = labels_r.iloc[ts:te]
        ret_te = daily_ret.iloc[ts:te]
        best_sh, best_t = -999.0, THRESH[0]
        for bt, nt in THRESH:
            w  = lab_te.map({0: bt, 3: bt, 1: nt, 2: nt, 4: 0.0, 5: 0.0}).fillna(0.0)
            r  = w.shift(1).fillna(0) * ret_te
            sh = _sharpe(r.fillna(0).values)
            if sh > best_sh:
                best_sh, best_t = sh, (bt, nt)
        wf_folds.append({
            'fold': fold + 1,
            'test_start': daily_ret.index[ts].strftime('%Y-%m-%d'),
            'test_end':   daily_ret.index[te - 1].strftime('%Y-%m-%d'),
            'best_sharpe': _safe(best_sh),
            'bull_thresh': best_t[0],
            'neutral_thresh': best_t[1],
        })
    wf_shs = [f['best_sharpe'] for f in wf_folds if f['best_sharpe'] is not None]
    wf_out = {
        'folds':       wf_folds,
        'sharpe_mean': _safe(float(np.mean(wf_shs))) if wf_shs else None,
        'sharpe_std':  _safe(float(np.std(wf_shs)))  if wf_shs else None,
    }

    # ── 11. Cross-Validation (IC stability) ───────────────────────────────────
    tss    = TimeSeriesSplit(n_splits=5)
    w_arr  = weights.values
    fv_arr = fwd_ret.fillna(0).values
    cv_ics = []
    for _, te_idx in tss.split(w_arr):
        w_te  = w_arr[te_idx]
        f_te  = fv_arr[te_idx]
        mask  = ~np.isnan(f_te) & ~np.isnan(w_te)
        if mask.sum() > 10:
            rho, _ = spearmanr(w_te[mask], f_te[mask])
            cv_ics.append(0.0 if math.isnan(float(rho)) else float(rho))
        else:
            cv_ics.append(0.0)
    cv_out = {
        'fold_ics': [round(v, 4) for v in cv_ics],
        'ic_mean':  _safe(float(np.mean(cv_ics))),
        'ic_std':   _safe(float(np.std(cv_ics))),
    }

    # ── Assemble final response ───────────────────────────────────────────────
    return {
        'ticker':     ticker.upper(),
        'start_date': start_date,
        'end_date':   end_date,
        'benchmark':  BENCHMARK_TICKER,
        'n_trading_days': n_days,

        'performance': {
            'sharpe_ratio':  _safe(sharpe_s),
            'sortino_ratio': _safe(sortino_s),
            'calmar_ratio':  _safe(calmar_s),
            'max_drawdown':  _safe(mdd_s),
            'cagr':          _safe(cagr_s),
            'alpha':         _safe(alpha),
            'beta':          _safe(beta),
            'sharpe_bh':     _safe(sharpe_bh),
            'mdd_bh':        _safe(mdd_bh),
            'cagr_bh':       _safe(cagr_bh),
        },

        'ic': {
            'mean':       _safe(ic_mean),
            'std':        _safe(ic_std),
            'ir':         _safe(ic_ir),
            'series':     [round(v, 4) for v in ic_vals[-200:]],
            'dates':      ic_dates[-200:],
            't_stat':     _safe(t_stat),
            'p_value':    _safe(p_val),
            'significant': ic_sig,
        },

        'rolling': rolling_out,
        'monte_carlo': mc_out,
        'transition_matrix': tm_out,
        'feature_importance': fi_out,

        'regime_confidence': {
            'current':   round(float(conf[-1]), 3),
            'by_regime': conf_by,
        },

        'walk_forward':      wf_out,
        'cross_validation':  cv_out,
    }
