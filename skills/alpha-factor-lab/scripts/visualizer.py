#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块 v2 (Visualizer)

功能：
  - 分层净值曲线
  - IC 时序 + 累计 IC（正负着色 + 滚动均值）
  - 因子分布直方图（含偏度/峰度统计）
  - 因子自相关衰减图
  - 回测摘要四宫格

v2 改进：
  - 使用英文标签避免中文字体问题
  - IC 图增加 Newey-West 显著性标注
  - 回测摘要增加 Calmar ratio
  - 因子分布增加 QQ 图

用法：
  python3 visualizer.py \\
    --backtest-report backtest_report.json \\
    --decay-report decay_report.json \\
    --factor factor_values.csv \\
    --output-dir output/
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 配色方案
QUINTILE_COLORS = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"]
LS_COLOR = "#9467bd"
COLORS = QUINTILE_COLORS + [LS_COLOR, "#8c564b", "#e377c2", "#7f7f7f", "#17becf"]


def setup_style():
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


setup_style()


def plot_quintile_returns(data_path: str, output_path: str,
                           factor_name: str = "factor"):
    """分层净值曲线。"""
    with open(data_path) as f:
        data = json.load(f)
    
    dates = pd.to_datetime(data["dates"])
    series = data["series"]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n_groups = sum(1 for k in series if k.startswith("group_"))
    for key in sorted(k for k in series if k.startswith("group_")):
        gnum = int(key.split("_")[1])
        values = series[key]
        color = QUINTILE_COLORS[(gnum - 1) % len(QUINTILE_COLORS)]
        lw = 2.0 if gnum in (1, n_groups) else 1.0
        suffix = " (Low)" if gnum == 1 else " (High)" if gnum == n_groups else ""
        ax.plot(dates[:len(values)], values, color=color,
                linewidth=lw, label=f"Q{gnum}{suffix}", alpha=0.85)
    
    if "long_short" in series:
        values = series["long_short"]
        ax.plot(dates[:len(values)], values, color=LS_COLOR,
                linewidth=2.5, label="Long-Short", linestyle="--")
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return (NAV)", fontsize=12)
    ax.set_title(f"Quintile Returns: {factor_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.8)
    ax.axhline(y=1, color="gray", linewidth=0.5)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def plot_ic_series(data_path: str, output_path: str,
                    factor_name: str = "factor",
                    report_path: str = None):
    """IC 时序 + 累计 IC + 显著性标注。"""
    with open(data_path) as f:
        data = json.load(f)
    
    dates = pd.to_datetime(data["dates"])
    ic = np.array(data["ic"])
    
    # 从报告中读取 Newey-West 结果
    nw_info = ""
    if report_path and Path(report_path).exists():
        with open(report_path) as f:
            rpt = json.load(f)
        m = rpt.get("metrics", {})
        t_stat = m.get("ic_t_stat")
        sig5 = m.get("ic_significant_5pct", False)
        if t_stat is not None:
            nw_info = f"\nNW t={t_stat:.2f} {'✓' if sig5 else '✗'}@5%"
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10),
                              gridspec_kw={"height_ratios": [3, 2]})
    
    # IC 时序
    ax1 = axes[0]
    colors_bar = ["#2ca02c" if v >= 0 else "#d62728" for v in ic]
    ax1.bar(dates, ic, width=max(2, len(dates) // 100), color=colors_bar, alpha=0.6)
    
    ic_s = pd.Series(ic, index=dates)
    rolling_window = min(20, max(3, len(ic) // 5))
    rolling_ic = ic_s.rolling(rolling_window, min_periods=2).mean()
    ax1.plot(dates, rolling_ic, color="navy", linewidth=2,
             label=f"IC {rolling_window}d MA")
    
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_ylabel("IC", fontsize=12)
    ax1.set_title(f"Information Coefficient: {factor_name}", fontsize=14,
                  fontweight="bold")
    
    ic_mean = np.nanmean(ic)
    ic_std = np.nanstd(ic)
    ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_pos = np.nanmean(ic > 0)
    
    info = (f"IC Mean: {ic_mean:.4f}\nIC Std: {ic_std:.4f}\n"
            f"IR: {ir:.4f}\nIC>0: {ic_pos:.1%}{nw_info}")
    ax1.text(0.02, 0.95, info, transform=ax1.transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax1.legend(loc="upper right", fontsize=10)
    
    # 累计 IC
    ax2 = axes[1]
    cum_ic = np.nancumsum(ic)
    ax2.plot(dates, cum_ic, color="steelblue", linewidth=2)
    ax2.fill_between(dates, 0, cum_ic, alpha=0.2, color="steelblue")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Cumulative IC", fontsize=12)
    ax2.set_title("Cumulative IC", fontsize=12)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def plot_factor_distribution(factor_path: str, output_path: str,
                              factor_name: str = "factor"):
    """因子分布直方图 + QQ 图。"""
    df = pd.read_csv(factor_path, encoding="utf-8")
    factor_cols = [c for c in df.columns if c not in ("date", "stock_code")]
    if not factor_cols:
        return
    
    factor_col = factor_cols[0]
    values = df[factor_col].dropna()
    if len(values) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 直方图
    ax1 = axes[0]
    n_bins = min(100, max(20, len(values) // 100))
    ax1.hist(values, bins=n_bins, color="steelblue", alpha=0.7,
             edgecolor="white", linewidth=0.5)
    ax1.axvline(x=values.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean: {values.mean():.4f}")
    ax1.axvline(x=values.median(), color="orange", linestyle="--", linewidth=1.5,
                label=f"Median: {values.median():.4f}")
    ax1.set_xlabel("Factor Value", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title(f"Distribution: {factor_name}", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    
    info = (f"N: {len(values)}\nMean: {values.mean():.4f}\n"
            f"Std: {values.std():.4f}\nSkew: {values.skew():.4f}\n"
            f"Kurt: {values.kurtosis():.4f}")
    ax1.text(0.95, 0.95, info, transform=ax1.transAxes, fontsize=9,
             va="top", ha="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    # QQ 图（代替原来的时序截面均值图——更有诊断价值）
    from scipy import stats as sp_stats
    ax2 = axes[1]
    sample = values.sample(min(5000, len(values)), random_state=42).sort_values()
    theoretical = sp_stats.norm.ppf(
        np.linspace(0.001, 0.999, len(sample))
    )
    ax2.scatter(theoretical, sample.values, s=2, color="steelblue", alpha=0.5)
    # 45 度参考线
    lo = min(theoretical.min(), sample.values.min())
    hi = max(theoretical.max(), sample.values.max())
    ax2.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Normal")
    ax2.set_xlabel("Theoretical Quantiles", fontsize=12)
    ax2.set_ylabel("Sample Quantiles", fontsize=12)
    ax2.set_title("Q-Q Plot (vs Normal)", fontsize=12)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def plot_factor_decay(decay_report_path: str, output_path: str):
    """因子自相关衰减图。"""
    with open(decay_report_path) as f:
        report = json.load(f)
    
    factor_name = report.get("factor_name", "factor")
    autocorr = report.get("autocorrelations", {})
    decay_params = report.get("decay_params", {})
    
    if not autocorr:
        if report.get("static_factor"):
            print("[信息] 静态因子，跳过衰减图")
        return
    
    lags, means, stds = [], [], []
    for key, stats in sorted(autocorr.items(), key=lambda x: int(x[0])):
        if stats["mean"] is not None:
            lags.append(int(key))
            means.append(stats["mean"])
            stds.append(stats.get("std", 0))
    
    if not lags:
        return
    
    lags = np.array(lags)
    means = np.array(means)
    stds = np.array(stds)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(lags, means, alpha=0.6, color="steelblue", width=0.8,
           label="Rank Autocorrelation")
    ax.errorbar(lags, means, yerr=stds, fmt="none", ecolor="gray",
                alpha=0.5, capsize=2)
    
    if decay_params.get("fit_success"):
        x_fit = np.linspace(0, max(lags) + 1, 200)
        y_fit = (decay_params["a"] * np.exp(-x_fit / decay_params["tau"])
                 + decay_params["c"])
        ax.plot(x_fit, y_fit, "r-", linewidth=2.5, label="Exp Fit", alpha=0.8)
    
    hl = decay_params.get("half_life")
    if hl is not None:
        ax.axvline(x=hl, color="orange", linestyle="--", linewidth=2,
                   label=f"Half-life = {hl:.1f}d")
    
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (days)", fontsize=12)
    ax.set_ylabel("Rank Autocorrelation", fontsize=12)
    ax.set_title(f"Decay: {factor_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    
    info_parts = []
    if hl is not None:
        info_parts.append(f"Half-life: {hl:.1f}d")
    r2 = decay_params.get("r_squared")
    if r2 is not None:
        info_parts.append(f"R²: {r2:.3f}")
    if info_parts:
        ax.text(0.95, 0.75, "\n".join(info_parts), transform=ax.transAxes,
                fontsize=11, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def plot_backtest_summary(report_path: str, output_path: str):
    """回测摘要四宫格。"""
    with open(report_path) as f:
        report = json.load(f)
    
    factor_name = report.get("factor_name", "factor")
    metrics = report.get("metrics", {})
    n_groups = report.get("n_groups", 5)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Backtest Summary: {factor_name}", fontsize=16,
                 fontweight="bold", y=1.02)
    
    # 1. 分层年化收益
    ax1 = axes[0][0]
    group_rets = metrics.get("group_returns_annualized", [])
    if group_rets:
        x = range(1, len(group_rets) + 1)
        vals = [r * 100 if r is not None else 0 for r in group_rets]
        bars = ax1.bar(x, vals,
                       color=[QUINTILE_COLORS[i % len(QUINTILE_COLORS)]
                              for i in range(len(vals))],
                       alpha=0.8, edgecolor="white")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels([f"Q{i}" for i in x])
        ax1.set_ylabel("Annualized Return (%)")
        ax1.set_title("Quintile Returns (Annualized)")
        ax1.axhline(y=0, color="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    
    # 2. 指标表格
    ax2 = axes[0][1]
    ax2.axis("off")
    
    def fmt(v, pct=False):
        if v is None:
            return "N/A"
        return f"{v:.1%}" if pct else f"{v:.4f}"
    
    sig_icon = "✓" if metrics.get("ic_significant_5pct") else "✗"
    table_data = [
        ["IC Mean", fmt(metrics.get("ic_mean"))],
        ["Rank IC", fmt(metrics.get("rank_ic_mean"))],
        ["IR", fmt(metrics.get("ir"))],
        ["IC > 0", fmt(metrics.get("ic_positive_pct"), pct=True)],
        ["NW t-stat", f"{metrics.get('ic_t_stat', 'N/A')} {sig_icon}"],
        ["L/S Ann. Return", fmt(metrics.get("long_short_ann_return"), pct=True)],
        ["L/S Sharpe", fmt(metrics.get("long_short_sharpe"))],
        ["L/S MDD", fmt(metrics.get("long_short_mdd"), pct=True)],
        ["Turnover", fmt(metrics.get("turnover_mean"), pct=True)],
        ["Monotonicity", fmt(metrics.get("monotonicity"))],
    ]
    
    table = ax2.table(cellText=table_data, colLabels=["Metric", "Value"],
                      loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.7)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D6E4F0")
    ax2.set_title("Key Metrics", fontsize=12, pad=20)
    
    # 3. Sharpe 柱状图
    ax3 = axes[1][0]
    gm = metrics.get("group_metrics", {})
    if gm:
        labels, sharpes = [], []
        for g in range(1, n_groups + 1):
            key = f"group_{g}"
            if key in gm:
                labels.append(f"Q{g}")
                sharpes.append(gm[key].get("sharpe_ratio", 0) or 0)
        if "long_short" in gm:
            labels.append("L/S")
            sharpes.append(gm["long_short"].get("sharpe_ratio", 0) or 0)
        
        bar_colors = [QUINTILE_COLORS[i % len(QUINTILE_COLORS)]
                      for i in range(len(labels))]
        if len(labels) > n_groups:
            bar_colors[-1] = LS_COLOR
        
        ax3.bar(labels, sharpes, color=bar_colors, alpha=0.8)
        ax3.set_ylabel("Sharpe Ratio")
        ax3.set_title("Sharpe by Group")
        ax3.axhline(y=0, color="black", linewidth=0.5)
    
    # 4. MDD 柱状图
    ax4 = axes[1][1]
    if gm:
        labels, mdds = [], []
        for g in range(1, n_groups + 1):
            key = f"group_{g}"
            if key in gm:
                labels.append(f"Q{g}")
                mdds.append(abs(gm[key].get("max_drawdown", 0) or 0) * 100)
        if "long_short" in gm:
            labels.append("L/S")
            mdds.append(abs(gm["long_short"].get("max_drawdown", 0) or 0) * 100)
        
        ax4.bar(labels, mdds, color="#d62728", alpha=0.7)
        ax4.set_ylabel("Max Drawdown (%)")
        ax4.set_title("Max Drawdown by Group")
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="因子研究可视化 v2")
    parser.add_argument("--backtest-report", default=None)
    parser.add_argument("--decay-report", default=None)
    parser.add_argument("--factor", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--factor-name", default="factor")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n = 0
    
    if args.backtest_report:
        bt_dir = Path(args.backtest_report).parent
        
        # 中间数据可能在 backtest-report 同目录或 output-dir 下
        search_dirs = [bt_dir, output_dir]
        
        cum_path = None
        for d in search_dirs:
            p = d / "cumulative_returns.json"
            if p.exists():
                cum_path = p
                break
        if cum_path:
            plot_quintile_returns(str(cum_path),
                                  str(output_dir / "quintile_returns.png"),
                                  args.factor_name)
            n += 1
        
        ic_path = None
        for d in search_dirs:
            p = d / "ic_series.json"
            if p.exists():
                ic_path = p
                break
        if ic_path:
            plot_ic_series(str(ic_path),
                           str(output_dir / "ic_series.png"),
                           args.factor_name,
                           args.backtest_report)
            n += 1
        
        plot_backtest_summary(args.backtest_report,
                               str(output_dir / "backtest_summary.png"))
        n += 1
    
    if args.factor:
        plot_factor_distribution(args.factor,
                                  str(output_dir / "factor_distribution.png"),
                                  args.factor_name)
        n += 1
    
    if args.decay_report:
        plot_factor_decay(args.decay_report,
                           str(output_dir / "factor_decay.png"))
        n += 1
    
    print(f"\n[完成] {n} 张图 → {output_dir}")


if __name__ == "__main__":
    main()
