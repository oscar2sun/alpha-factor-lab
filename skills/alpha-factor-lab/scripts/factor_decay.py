#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子衰减分析 v2 (Factor Decay Analysis)

功能：
  - 计算因子值的截面自相关系数（lag-1 到 lag-N）
  - 拟合指数衰减曲线 autocorr(lag) = a * exp(-lag/tau) + c
  - 估算半衰期 half_life = tau * ln(2)
  - 自动检测静态因子并给出警告
  - 输出衰减报告（JSON）和图表（PNG）

v2 改进：
  - 自动检测静态因子，对静态因子给出警告（自相关恒 ≈ 1）
  - 拟合边界 tau 上限动态调整为 5 × max_lag
  - 增加 IC 衰减分析模式（--ic-decay）：分析不同前瞻窗口下的 IC
  - 改进半衰期的 fallback 估算（分段线性插值）

用法：
  python3 factor_decay.py \\
    --factor factor_values.csv \\
    --max-lag 20 \\
    --output-report decay_report.json \\
    --output-chart decay_chart.png

  # IC 衰减模式
  python3 factor_decay.py \\
    --factor factor_values.csv \\
    --returns returns.csv \\
    --ic-decay \\
    --max-lag 60 \\
    --output-report ic_decay_report.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")


def load_factor_matrix(filepath: str) -> tuple:
    """加载因子值矩阵（pivot 成 date × stock）。"""
    df = pd.read_csv(filepath, encoding="utf-8")
    
    if "date" not in df.columns or "stock_code" not in df.columns:
        print(f"[错误] 因子文件必须包含 'date' 和 'stock_code' 列", file=sys.stderr)
        sys.exit(1)
    
    factor_cols = [c for c in df.columns if c not in ("date", "stock_code")]
    if not factor_cols:
        print(f"[错误] 因子文件无数据列", file=sys.stderr)
        sys.exit(1)
    
    factor_col = factor_cols[0]
    print(f"[信息] 因子列: {factor_col}")
    
    df["date"] = pd.to_datetime(df["date"])
    matrix = df.pivot_table(index="date", columns="stock_code", values=factor_col)
    matrix = matrix.sort_index()
    
    return matrix, factor_col


def is_static_factor(matrix: pd.DataFrame, threshold: float = 0.99) -> bool:
    """检测因子是否为静态。"""
    stds = matrix.std(axis=0)
    return (stds < 1e-10).mean() >= threshold


def compute_cross_sectional_autocorrelation(matrix: pd.DataFrame,
                                             max_lag: int) -> dict:
    """
    计算因子截面排名自相关。
    
    Lag k: corr(rank_t, rank_{t+k})，取时序均值。
    衡量因子排名的持续性。
    """
    dates = matrix.index.tolist()
    n_dates = len(dates)
    autocorr_results = {}
    
    for lag in range(1, max_lag + 1):
        correlations = []
        for i in range(n_dates - lag):
            current = matrix.iloc[i]
            future = matrix.iloc[i + lag]
            valid = current.notna() & future.notna()
            if valid.sum() < 5:
                continue
            
            corr = current[valid].rank().corr(future[valid].rank())
            if not np.isnan(corr):
                correlations.append(corr)
        
        autocorr_results[lag] = {
            "mean": float(np.mean(correlations)) if correlations else np.nan,
            "std": float(np.std(correlations)) if correlations else np.nan,
            "count": len(correlations),
            "median": float(np.median(correlations)) if correlations else np.nan,
        }
    
    return autocorr_results


def compute_ic_decay(factor_matrix: pd.DataFrame,
                      return_matrix: pd.DataFrame,
                      max_lag: int = 60,
                      step: int = 5) -> dict:
    """
    IC 衰减分析：不同前瞻窗口下的 IC。
    
    对于 forward_days = [1, step, 2*step, ..., max_lag]，
    计算每个窗口的 IC 均值。
    
    衡量因子信号在多长时间内有预测力。
    """
    from factor_backtest import compute_forward_returns
    
    common_stocks = factor_matrix.columns.intersection(return_matrix.columns)
    common_dates = factor_matrix.index.intersection(return_matrix.index)
    
    # 取因子的截面均值（处理静态因子）
    factor_avg = factor_matrix.loc[common_dates, common_stocks].mean(axis=0)
    
    windows = list(range(1, max_lag + 1, step))
    if 1 not in windows:
        windows = [1] + windows
    
    ic_decay = {}
    for fwd in windows:
        forward_ret = compute_forward_returns(return_matrix[common_stocks], fwd)
        
        # 不重叠评估
        eval_dates = common_dates[::max(fwd, 1)]
        ic_vals = []
        for date in eval_dates:
            if date not in forward_ret.index:
                continue
            r = forward_ret.loc[date, common_stocks]
            valid = factor_avg.notna() & r.notna()
            if valid.sum() < 5:
                continue
            corr, _ = sp_stats.spearmanr(factor_avg[valid], r[valid])
            if not np.isnan(corr):
                ic_vals.append(corr)
        
        ic_decay[fwd] = {
            "ic_mean": float(np.mean(ic_vals)) if ic_vals else np.nan,
            "ic_std": float(np.std(ic_vals)) if ic_vals else np.nan,
            "count": len(ic_vals),
        }
    
    return ic_decay


def fit_exponential_decay(autocorr_results: dict,
                           max_lag: int = 20) -> dict:
    """
    拟合 f(lag) = a * exp(-lag/tau) + c。
    tau 上限动态调整为 5 × max_lag。
    """
    lags, values = [], []
    for lag, stats in autocorr_results.items():
        if not np.isnan(stats["mean"]):
            lags.append(lag)
            values.append(stats["mean"])
    
    if len(lags) < 3:
        return _fallback_half_life(lags, values)
    
    lags = np.array(lags, dtype=float)
    values = np.array(values, dtype=float)
    
    def decay_func(x, a, tau, c):
        return a * np.exp(-x / tau) + c
    
    try:
        a0 = max(values[0] - values[-1], 0.01)
        tau0 = max(min(lags[-1] / 3, 99), 0.2)
        c0 = max(min(values[-1], 0.99), -0.99)
        tau_upper = 5 * max_lag  # 动态上限
        
        popt, _ = curve_fit(
            decay_func, lags, values,
            p0=[a0, tau0, c0],
            bounds=([0, 0.1, -1], [2, tau_upper, 1]),
            maxfev=5000,
        )
        
        a, tau, c = popt
        half_life = tau * np.log(2)
        
        y_pred = decay_func(lags, *popt)
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            "tau": float(tau),
            "half_life": float(half_life),
            "a": float(a),
            "c": float(c),
            "r_squared": float(r_squared),
            "fit_success": True,
        }
    
    except Exception as e:
        print(f"[警告] 拟合失败: {e}", file=sys.stderr)
        return _fallback_half_life(lags, values)


def _fallback_half_life(lags, values):
    """线性插值估算半衰期。"""
    lags = np.array(lags, dtype=float) if not isinstance(lags, np.ndarray) else lags
    values = np.array(values, dtype=float) if not isinstance(values, np.ndarray) else values
    
    if len(values) < 2:
        return {"tau": np.nan, "half_life": np.nan, "a": np.nan,
                "c": np.nan, "r_squared": np.nan, "fit_success": False}
    
    half_val = values[0] / 2
    half_life_est = np.nan
    for i in range(1, len(values)):
        if values[i] < half_val:
            # 分段线性插值
            half_life_est = (
                lags[i - 1]
                + (half_val - values[i - 1])
                / (values[i] - values[i - 1])
                * (lags[i] - lags[i - 1])
            )
            break
    
    tau = half_life_est / np.log(2) if not np.isnan(half_life_est) else np.nan
    return {
        "tau": float(tau) if not np.isnan(tau) else np.nan,
        "half_life": float(half_life_est) if not np.isnan(half_life_est) else np.nan,
        "a": np.nan, "c": np.nan, "r_squared": np.nan, "fit_success": False,
    }


def generate_decay_chart(autocorr_results: dict, decay_params: dict,
                          output_path: str, factor_name: str = "factor"):
    """生成自相关衰减图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    lags, means, stds = [], [], []
    for lag, stats in sorted(autocorr_results.items()):
        if not np.isnan(stats["mean"]):
            lags.append(lag)
            means.append(stats["mean"])
            stds.append(stats["std"])
    
    if not lags:
        print("[警告] 无数据，跳过图表", file=sys.stderr)
        return
    
    lags = np.array(lags)
    means = np.array(means)
    stds = np.array(stds)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(lags, means, alpha=0.6, color="steelblue", label="Rank Autocorrelation")
    ax.errorbar(lags, means, yerr=stds, fmt="none", ecolor="gray", alpha=0.5, capsize=2)
    
    if decay_params.get("fit_success"):
        x_fit = np.linspace(0, max(lags) + 1, 200)
        y_fit = (decay_params["a"] * np.exp(-x_fit / decay_params["tau"])
                 + decay_params["c"])
        ax.plot(x_fit, y_fit, "r-", linewidth=2.5, label="Exponential Fit", alpha=0.8)
    
    half_life = decay_params.get("half_life")
    if half_life and not np.isnan(half_life):
        ax.axvline(x=half_life, color="orange", linestyle="--", linewidth=2,
                   label=f"Half-life = {half_life:.1f} days")
    
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (trading days)", fontsize=12)
    ax.set_ylabel("Cross-sectional Rank Autocorrelation", fontsize=12)
    ax.set_title(f"Factor Decay: {factor_name}", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    
    info = ""
    if half_life and not np.isnan(half_life):
        info += f"Half-life: {half_life:.1f} days\n"
    r2 = decay_params.get("r_squared")
    if r2 and not np.isnan(r2):
        info += f"R²: {r2:.3f}"
    if info:
        ax.text(0.95, 0.95, info, transform=ax.transAxes,
                fontsize=10, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[信息] 衰减图: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="因子衰减分析 v2")
    parser.add_argument("--factor", required=True, help="因子值 CSV")
    parser.add_argument("--max-lag", type=int, default=20, help="最大滞后阶数")
    parser.add_argument("--output-report", required=True, help="输出 JSON")
    parser.add_argument("--output-chart", default=None, help="输出 PNG")
    parser.add_argument("--returns", default=None,
                        help="收益率 CSV（IC 衰减模式需要）")
    parser.add_argument("--ic-decay", action="store_true",
                        help="IC 衰减模式（分析不同前瞻窗口的 IC）")
    parser.add_argument("--ic-step", type=int, default=5,
                        help="IC 衰减步长（默认 5 天）")
    
    args = parser.parse_args()
    
    print(f"[信息] 加载因子: {args.factor}")
    matrix, factor_name = load_factor_matrix(args.factor)
    print(f"[信息] 矩阵: {matrix.shape[0]} 日 × {matrix.shape[1]} 股")
    
    # 静态因子检测
    static = is_static_factor(matrix)
    if static:
        print(f"[⚠ 警告] 检测到静态因子！")
        print(f"  静态因子的截面排名不变，自相关恒 ≈ 1，衰减分析无意义。")
        print(f"  建议使用 --ic-decay 模式分析因子信号的持续性。")
    
    report = {
        "factor_name": factor_name,
        "n_dates": int(matrix.shape[0]),
        "n_stocks": int(matrix.shape[1]),
        "max_lag": args.max_lag,
        "static_factor": static,
    }
    
    # IC 衰减模式
    if args.ic_decay:
        if not args.returns:
            print("[错误] IC 衰减模式需要 --returns", file=sys.stderr)
            sys.exit(1)
        
        from factor_backtest import load_matrix as load_ret_matrix
        ret_matrix, _ = load_ret_matrix(args.returns)
        
        print(f"[信息] IC 衰减分析 (step={args.ic_step})...")
        ic_decay = compute_ic_decay(matrix, ret_matrix, args.max_lag, args.ic_step)
        report["ic_decay"] = ic_decay
        
        print(f"\n[结果] IC 衰减:")
        for fwd, stats in sorted(ic_decay.items()):
            print(f"  Forward {fwd:3d}d: IC={stats['ic_mean']:>7.4f} "
                  f"± {stats['ic_std']:>6.4f} (N={stats['count']})")
    
    # 标准自相关衰减
    if not static:
        print(f"[信息] 计算自相关 (max_lag={args.max_lag})...")
        autocorr = compute_cross_sectional_autocorrelation(matrix, args.max_lag)
        
        print(f"[信息] 拟合衰减曲线...")
        decay_params = fit_exponential_decay(autocorr, args.max_lag)
        
        report["decay_params"] = decay_params
        report["autocorrelations"] = {
            str(k): {"lag": k, "mean": v["mean"], "std": v["std"],
                     "median": v["median"], "count": v["count"]}
            for k, v in autocorr.items()
        }
        
        if args.output_chart:
            generate_decay_chart(autocorr, decay_params, args.output_chart, factor_name)
        
        half_life = decay_params.get("half_life", np.nan)
        print(f"\n[结果] 半衰期: {half_life:.1f} 天")
        print(f"[结果] R²: {decay_params.get('r_squared', np.nan):.3f}")
        
        if not np.isnan(half_life):
            if half_life > 20:
                rating = "优秀 ★★★ — 信号持续性强"
            elif half_life > 10:
                rating = "良好 ★★☆ — 有一定持续性"
            elif half_life > 5:
                rating = "一般 ★☆☆ — 衰减较快"
            else:
                rating = "较差 ☆☆☆ — 需高频调仓"
            print(f"[评级] {rating}")
        
        print(f"\n[详情] 前 10 lag:")
        for lag in sorted(autocorr.keys())[:10]:
            s = autocorr[lag]
            print(f"  Lag {lag:2d}: {s['mean']:.4f} ± {s['std']:.4f}")
    else:
        print(f"\n[信息] 跳过自相关分析（静态因子）")
        report["decay_params"] = {
            "tau": np.nan, "half_life": np.nan,
            "note": "Static factor — autocorrelation ≈ 1 at all lags, use --ic-decay instead",
            "fit_success": False,
        }
    
    # 保存
    def nan_to_none(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: nan_to_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [nan_to_none(v) for v in obj]
        return obj
    
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(nan_to_none(report), f, indent=2, ensure_ascii=False)
    print(f"[信息] 报告: {args.output_report}")


if __name__ == "__main__":
    main()
