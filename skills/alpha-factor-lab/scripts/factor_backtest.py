#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单因子回测引擎 v2 (Single Factor Backtest Engine)

功能：
  - 因子分层排序（quintile sort）
  - 各分层组合等权收益计算
  - 多空组合（Long-Short）收益
  - IC / Rank IC / IR（支持多期前瞻收益）
  - Sharpe / MDD / Turnover / 单调性
  - 区分静态因子 vs 动态因子的 IC 计算
  - 可选交易成本扣除
  - Newey-West t 检验判断 IC 显著性
  - 输出回测报告（JSON）和中间数据（CSV/JSON）

v2 改进（相比 v1）：
  - 修正 IC 计算: 静态因子用不重叠窗口，避免自相关膨胀
  - 修正前瞻收益: 正确使用 t+1 到 t+N 的累计收益
  - 新增 --forward-days 控制前瞻窗口
  - 新增 --cost 交易成本参数
  - 新增 --static-factor 标志位（自动检测 or 手动指定）
  - 年化收益计算修正（处理负收益溢出）
  - 增加 Newey-West t 检验 + Calmar ratio

用法：
  python3 factor_backtest.py \\
    --factor factor_values.csv \\
    --returns returns.csv \\
    --n-groups 5 \\
    --rebalance-freq 20 \\
    --forward-days 20 \\
    --cost 0.002 \\
    --output-report backtest_report.json \\
    --output-dir output/
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")


# ============================================================
# 数据加载
# ============================================================

def load_matrix(filepath: str, value_col: str = None) -> tuple:
    """加载 CSV 并 pivot 为 (date × stock) 矩阵。"""
    df = pd.read_csv(filepath, encoding="utf-8")
    
    if "date" not in df.columns or "stock_code" not in df.columns:
        print(f"[错误] 文件 {filepath} 必须包含 'date' 和 'stock_code' 列",
              file=sys.stderr)
        sys.exit(1)
    
    data_cols = [c for c in df.columns if c not in ("date", "stock_code")]
    if not data_cols:
        print(f"[错误] 文件 {filepath} 没有数据列", file=sys.stderr)
        sys.exit(1)
    
    col = value_col if value_col and value_col in data_cols else data_cols[0]
    df["date"] = pd.to_datetime(df["date"])
    matrix = df.pivot_table(index="date", columns="stock_code", values=col)
    matrix = matrix.sort_index()
    
    return matrix, col


def compute_returns_from_prices(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """从价格矩阵计算日收益率。"""
    return price_matrix.pct_change()


# ============================================================
# 前瞻收益 & IC 计算
# ============================================================

def compute_forward_returns(return_matrix: pd.DataFrame,
                             forward_days: int = 20) -> pd.DataFrame:
    """
    计算前瞻 N 日累计收益。
    
    forward_return(t) = 从 t+1 到 t+N 的累计收益
                      = prod(1 + r_{t+1}, ..., 1 + r_{t+N}) - 1
    
    使用对数累计相减法，精确且高效。
    """
    log_ret = np.log1p(return_matrix.clip(lower=-0.999))  # 防止 log(0)
    cum_log = log_ret.cumsum()
    # forward(t) = cumlog(t + N) - cumlog(t)
    forward_cum = cum_log.shift(-forward_days) - cum_log
    return np.expm1(forward_cum)


def compute_ic_dynamic(factor_matrix: pd.DataFrame,
                        return_matrix: pd.DataFrame,
                        forward_days: int = 20,
                        method: str = "pearson") -> pd.Series:
    """
    动态因子的 IC 计算（不重叠窗口）。
    
    每隔 forward_days 天取一个评估点，计算截面 IC：
    IC(t) = corr(factor(t), forward_return(t, t+N))
    
    使用不重叠窗口避免前瞻收益重叠导致的 IC 自相关膨胀，
    保证 IC 序列近似独立，使 IR 和 t 检验统计量可靠。
    """
    forward_ret = compute_forward_returns(return_matrix, forward_days)
    common_dates = sorted(factor_matrix.index.intersection(forward_ret.index))
    common_stocks = factor_matrix.columns.intersection(forward_ret.columns)
    
    # 不重叠采样：每隔 forward_days 取一个评估点
    eval_dates = common_dates[::forward_days]
    
    ic_values = {}
    for date in eval_dates:
        f = factor_matrix.loc[date, common_stocks]
        r = forward_ret.loc[date, common_stocks]
        valid = f.notna() & r.notna()
        if valid.sum() < 5:
            continue
        
        fv, rv = f[valid], r[valid]
        if method == "spearman":
            corr, _ = sp_stats.spearmanr(fv, rv)
        else:
            corr, _ = sp_stats.pearsonr(fv, rv)
        if not np.isnan(corr):
            ic_values[date] = corr
    
    return pd.Series(ic_values)


def compute_ic_static(factor_series: pd.Series,
                       return_matrix: pd.DataFrame,
                       eval_dates: list = None,
                       forward_days: int = 20,
                       method: str = "pearson") -> pd.Series:
    """
    静态因子的 IC 计算。
    
    因子值不随时间变化，为避免前瞻收益重叠导致 IC 自相关膨胀，
    每隔 forward_days 天取一个不重叠的评估点。
    """
    forward_ret = compute_forward_returns(return_matrix, forward_days)
    common_stocks = factor_series.index.intersection(forward_ret.columns)
    
    if eval_dates is None:
        all_dates = forward_ret.index.tolist()
        eval_dates = all_dates[::forward_days]
    
    ic_values = {}
    for date in eval_dates:
        if date not in forward_ret.index:
            continue
        r = forward_ret.loc[date, common_stocks]
        f = factor_series[common_stocks]
        valid = f.notna() & r.notna()
        if valid.sum() < 5:
            continue
        
        fv, rv = f[valid], r[valid]
        if method == "spearman":
            corr, _ = sp_stats.spearmanr(fv, rv)
        else:
            corr, _ = sp_stats.pearsonr(fv, rv)
        if not np.isnan(corr):
            ic_values[date] = corr
    
    return pd.Series(ic_values)


def newey_west_t_stat(ic_series: pd.Series, max_lags: int = None) -> dict:
    """
    Newey-West t 检验，修正自相关后检验 IC 均值是否显著≠0。
    """
    n = len(ic_series)
    if n < 3:
        return {"t_stat": np.nan, "p_value": np.nan, "significant_5pct": False}
    
    if max_lags is None:
        max_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    max_lags = min(max_lags, n - 2)
    
    ic = ic_series.values
    mean_ic = ic.mean()
    demeaned = ic - mean_ic
    
    gamma_0 = (demeaned ** 2).mean()
    nw_var = gamma_0
    for lag in range(1, max_lags + 1):
        weight = 1 - lag / (max_lags + 1)  # Bartlett kernel
        gamma_lag = (demeaned[lag:] * demeaned[:-lag]).mean()
        nw_var += 2 * weight * gamma_lag
    
    nw_var = max(nw_var, 1e-12)
    se = np.sqrt(nw_var / n)
    t_stat = mean_ic / se if se > 0 else 0
    p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=n - 1))
    
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant_5pct": p_value < 0.05,
        "significant_1pct": p_value < 0.01,
    }


# ============================================================
# 检测静态因子
# ============================================================

def is_static_factor(factor_matrix: pd.DataFrame, threshold: float = 0.99) -> bool:
    """
    自动检测因子是否为静态。
    如果超过 threshold 比例的股票因子值标准差 ≈ 0，则为静态。
    """
    stds = factor_matrix.std(axis=0)
    zero_std_ratio = (stds < 1e-10).mean()
    return zero_std_ratio >= threshold


# ============================================================
# 分层回测
# ============================================================

def compute_group_returns(
    factor_matrix: pd.DataFrame,
    return_matrix: pd.DataFrame,
    n_groups: int = 5,
    rebalance_freq: int = 20,
    cost_per_trade: float = 0.0,
    limit_filter: pd.DataFrame = None,
) -> tuple:
    """
    因子分层回测。
    
    调仓日按因子值排序分组，等权持有到下一调仓日。
    可选扣除交易成本（双边 cost * 换手比例）。
    可选涨跌停/停牌过滤（limit_filter 标记不可交易的股票）。
    
    返回: (group_returns_dict, turnovers_list, holdings_info)
      holdings_info: 各组每期有效持仓数，用于诊断
    """
    common_dates = sorted(factor_matrix.index.intersection(return_matrix.index))
    common_stocks = sorted(factor_matrix.columns.intersection(return_matrix.columns))
    
    factor_aligned = factor_matrix.loc[common_dates, common_stocks]
    return_aligned = return_matrix.loc[common_dates, common_stocks]
    
    rebalance_indices = set(range(0, len(common_dates), max(rebalance_freq, 1)))
    
    group_returns = {g: [] for g in range(1, n_groups + 1)}
    group_dates = []
    turnovers = []
    holdings_counts = {g: [] for g in range(1, n_groups + 1)}
    current_groups = None
    prev_groups = None
    
    for idx in range(len(common_dates) - 1):
        date = common_dates[idx]
        next_date = common_dates[idx + 1]
        
        # 调仓
        if idx in rebalance_indices or current_groups is None:
            factor_vals = factor_aligned.loc[date].dropna()
            
            # 涨跌停/停牌过滤：排除不可交易股票
            if limit_filter is not None and date in limit_filter.index:
                tradable = limit_filter.loc[date]
                tradable_stocks = tradable[tradable].index
                factor_vals = factor_vals[
                    factor_vals.index.intersection(tradable_stocks)
                ]
            
            if len(factor_vals) < n_groups:
                for g in range(1, n_groups + 1):
                    group_returns[g].append(0.0)
                    holdings_counts[g].append(0)
                group_dates.append(next_date)
                continue
            
            prev_groups = current_groups
            current_groups = {}
            ranks = factor_vals.rank(method="first")
            group_size = len(ranks) / n_groups
            
            for g in range(1, n_groups + 1):
                lower = (g - 1) * group_size
                upper = g * group_size
                members = ranks[(ranks > lower) & (ranks <= upper)].index
                current_groups[g] = members
            
            # 标准换手率: Σ|w_new - w_old| / 2
            # 等权组合简化为: (新增+移出的股票数) / (2 * 组内股票数)
            if prev_groups is not None:
                total_turnover = 0
                for g in range(1, n_groups + 1):
                    prev_set = set(prev_groups.get(g, []))
                    curr_set = set(current_groups.get(g, []))
                    n_curr = len(curr_set)
                    if n_curr > 0:
                        n_new = len(curr_set - prev_set)
                        n_out = len(prev_set - curr_set)
                        total_turnover += (n_new + n_out) / (2 * n_curr)
                turnovers.append(total_turnover / n_groups)
        
        # 各组等权日收益
        for g in range(1, n_groups + 1):
            members = current_groups.get(g, pd.Index([]))
            if len(members) > 0:
                ret = return_aligned.loc[next_date, members]
                valid_ret = ret.dropna()
                n_valid = len(valid_ret)
                holdings_counts[g].append(n_valid)
                if n_valid > 0:
                    daily_ret = float(valid_ret.mean())
                    # 调仓日扣成本
                    if (idx in rebalance_indices and prev_groups is not None
                            and cost_per_trade > 0):
                        prev_set = set(prev_groups.get(g, []))
                        curr_set = set(members)
                        n_total = len(curr_set)
                        n_changed = len(prev_set - curr_set) + len(curr_set - prev_set)
                        if n_total > 0:
                            cost = 2 * cost_per_trade * (n_changed / (2 * n_total))
                            daily_ret -= cost
                    group_returns[g].append(daily_ret)
                else:
                    group_returns[g].append(0.0)
            else:
                group_returns[g].append(0.0)
                holdings_counts[g].append(0)
        
        group_dates.append(next_date)
    
    result = {}
    for g in range(1, n_groups + 1):
        result[g] = pd.Series(group_returns[g], index=group_dates)
    result["long_short"] = result[n_groups] - result[1]
    
    # 持仓统计
    holdings_info = {}
    for g in range(1, n_groups + 1):
        counts = np.array(holdings_counts[g])
        holdings_info[g] = {
            "mean": float(counts.mean()) if len(counts) > 0 else 0,
            "min": int(counts.min()) if len(counts) > 0 else 0,
            "max": int(counts.max()) if len(counts) > 0 else 0,
        }
    
    return result, turnovers, holdings_info


# ============================================================
# 指标计算
# ============================================================

def safe_annualize(total_return: float, n_days: int,
                    annual_factor: float = 252) -> float:
    """安全年化收益，处理负收益溢出和极端幂次。"""
    if n_days < 1:
        return np.nan
    base = 1 + total_return
    if base <= 0:
        return -1.0
    exponent = annual_factor / n_days
    try:
        ann = base ** exponent - 1
        if np.isnan(ann) or np.isinf(ann):
            ann = np.exp(np.log(base) * exponent) - 1
        return float(ann)
    except (OverflowError, ValueError):
        return float(np.exp(np.log(max(base, 1e-10)) * exponent) - 1)


def compute_metrics(
    group_return_series: dict,
    ic_series: pd.Series,
    rank_ic_series: pd.Series,
    turnovers: list,
    n_groups: int,
    annual_factor: float = 252,
    holdings_info: dict = None,
) -> dict:
    """计算完整回测指标。"""
    metrics = {}
    
    # IC
    if len(ic_series) > 0:
        metrics["ic_mean"] = float(ic_series.mean())
        metrics["ic_std"] = float(ic_series.std())
        metrics["ic_positive_pct"] = float((ic_series > 0).mean())
        metrics["ic_count"] = int(len(ic_series))
    else:
        metrics["ic_mean"] = np.nan
        metrics["ic_std"] = np.nan
        metrics["ic_positive_pct"] = np.nan
        metrics["ic_count"] = 0
    
    # Rank IC
    if len(rank_ic_series) > 0:
        metrics["rank_ic_mean"] = float(rank_ic_series.mean())
        metrics["rank_ic_std"] = float(rank_ic_series.std())
    else:
        metrics["rank_ic_mean"] = np.nan
        metrics["rank_ic_std"] = np.nan
    
    # IR
    metrics["ir"] = (
        metrics["ic_mean"] / metrics["ic_std"]
        if metrics.get("ic_std") and metrics["ic_std"] > 0
        else np.nan
    )
    
    # Newey-West t
    nw = newey_west_t_stat(ic_series)
    metrics["ic_t_stat"] = nw["t_stat"]
    metrics["ic_p_value"] = nw["p_value"]
    metrics["ic_significant_5pct"] = nw.get("significant_5pct", False)
    metrics["ic_significant_1pct"] = nw.get("significant_1pct", False)
    
    # 各组指标
    group_metrics = {}
    for key, returns in group_return_series.items():
        if len(returns) == 0:
            continue
        label = f"group_{key}" if isinstance(key, int) else str(key)
        
        cum = (1 + returns).cumprod()
        total_ret = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0
        n_days = len(returns)
        
        ann_ret = safe_annualize(total_ret, n_days, annual_factor)
        ann_vol = float(returns.std() * np.sqrt(annual_factor))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        peak = cum.expanding(min_periods=1).max()
        dd = (cum - peak) / peak
        mdd = float(dd.min())
        calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan
        
        group_metrics[label] = {
            "total_return": total_ret,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "calmar_ratio": calmar,
        }
    
    metrics["group_metrics"] = group_metrics
    
    # 多空
    if "long_short" in group_return_series:
        ls = group_return_series["long_short"]
        ls_cum = (1 + ls).cumprod()
        ls_total = float(ls_cum.iloc[-1] - 1)
        n_days = len(ls)
        metrics["long_short_total_return"] = ls_total
        metrics["long_short_ann_return"] = safe_annualize(ls_total, n_days, annual_factor)
        ls_vol = float(ls.std() * np.sqrt(annual_factor))
        metrics["long_short_sharpe"] = (
            metrics["long_short_ann_return"] / ls_vol if ls_vol > 0 else 0
        )
        peak = ls_cum.expanding(min_periods=1).max()
        dd = (ls_cum - peak) / peak
        metrics["long_short_mdd"] = float(dd.min())
    
    # 换手率
    metrics["turnover_mean"] = float(np.mean(turnovers)) if turnovers else np.nan
    metrics["turnover_std"] = float(np.std(turnovers)) if turnovers else np.nan
    
    # 分层年化收益
    metrics["group_returns_annualized"] = [
        group_metrics.get(f"group_{g}", {}).get("annualized_return", np.nan)
        for g in range(1, n_groups + 1)
    ]
    
    # 单调性
    returns_list = metrics["group_returns_annualized"]
    valid_returns = [(i, r) for i, r in enumerate(returns_list)
                     if r is not None and not np.isnan(r)]
    if len(valid_returns) >= 3:
        ranks, rets = zip(*valid_returns)
        corr, _ = sp_stats.spearmanr(ranks, rets)
        metrics["monotonicity"] = float(corr)
    else:
        metrics["monotonicity"] = np.nan
    
    # 持仓诊断
    if holdings_info:
        metrics["holdings_info"] = {}
        for g, info in holdings_info.items():
            metrics["holdings_info"][f"group_{g}"] = info
        
        # 检查各组持仓是否均衡
        means = [holdings_info[g]["mean"] for g in range(1, n_groups + 1)]
        if means:
            ratio = max(means) / max(min(means), 1)
            metrics["holdings_imbalance_ratio"] = float(ratio)
            if ratio > 1.5:
                metrics["holdings_warning"] = (
                    f"各组持仓不均衡 (max/min={ratio:.1f}x)，"
                    f"多空组合可能暴露市场 beta"
                )
    
    return metrics


# ============================================================
# 保存中间数据
# ============================================================

def save_backtest_data(group_return_series, ic_series, rank_ic_series, output_dir):
    """保存供可视化模块使用的中间数据。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cum_returns = {}
    dates = None
    for key, returns in group_return_series.items():
        label = f"group_{key}" if isinstance(key, int) else str(key)
        cum_returns[label] = ((1 + returns).cumprod()).values.tolist()
        if dates is None:
            dates = [
                d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                for d in returns.index
            ]
    
    with open(output_path / "cumulative_returns.json", "w") as f:
        json.dump({"dates": dates, "series": cum_returns}, f, indent=2)
    
    ic_data = {
        "dates": [
            d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            for d in ic_series.index
        ],
        "ic": ic_series.values.tolist(),
        "rank_ic": rank_ic_series.reindex(ic_series.index).fillna(0).values.tolist(),
    }
    with open(output_path / "ic_series.json", "w") as f:
        json.dump(ic_data, f, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="单因子回测引擎 v2")
    parser.add_argument("--factor", required=True,
                        help="因子值 CSV")
    parser.add_argument("--returns", required=True,
                        help="收益率 CSV（或价格 CSV + --is-price）")
    parser.add_argument("--n-groups", type=int, default=5,
                        help="分组数（默认 5）")
    parser.add_argument("--rebalance-freq", type=int, default=20,
                        help="调仓频率/天（默认 20）")
    parser.add_argument("--decay-report", default=None,
                        help="衰减分析报告 JSON，自动读取半衰期设置调仓周期和前瞻天数")
    parser.add_argument("--forward-days", type=int, default=20,
                        help="IC 前瞻天数（默认 20）")
    parser.add_argument("--cost", type=float, default=0.0,
                        help="单边交易成本（默认 0）")
    parser.add_argument("--output-report", required=True,
                        help="输出报告 JSON")
    parser.add_argument("--output-dir", default=None,
                        help="输出目录")
    parser.add_argument("--factor-name", default=None,
                        help="因子名称")
    parser.add_argument("--is-price", action="store_true",
                        help="--returns 是价格数据")
    parser.add_argument("--static-factor", action="store_true",
                        help="强制标记为静态因子")
    parser.add_argument("--limit-filter", default=None,
                        help="涨跌停/停牌过滤 CSV（列: date, stock_code, tradable=1/0）")
    
    args = parser.parse_args()
    
    # 从衰减报告自动设置调仓周期
    decay_half_life = None
    if args.decay_report:
        try:
            with open(args.decay_report, "r", encoding="utf-8") as f:
                decay_data = json.load(f)
            # 优先从 IC 衰减中提取最佳前瞻窗口
            ic_decay = decay_data.get("ic_decay")
            if ic_decay:
                # 找 |IC| 最大的前瞻窗口作为最佳持仓周期
                best_fwd = None
                best_ic = 0
                for fwd_str, stats in ic_decay.items():
                    ic_val = abs(stats.get("ic_mean", 0) or 0)
                    if ic_val > best_ic:
                        best_ic = ic_val
                        best_fwd = int(fwd_str)
                if best_fwd:
                    decay_half_life = best_fwd
                    print(f"[信息] 从 IC 衰减报告读取: 最佳前瞻窗口 = {best_fwd}d (|IC|={best_ic:.4f})")
            # 否则用自相关半衰期
            if decay_half_life is None:
                dp = decay_data.get("decay_params", {})
                hl = dp.get("half_life")
                if hl is not None and not (isinstance(hl, float) and np.isnan(hl)):
                    decay_half_life = max(int(round(hl)), 1)
                    print(f"[信息] 从衰减报告读取: 半衰期 = {hl:.1f}d → 调仓周期 = {decay_half_life}d")
            if decay_half_life is None:
                print(f"[警告] 衰减报告中无有效半衰期，使用默认调仓周期 {args.rebalance_freq}d")
        except Exception as e:
            print(f"[警告] 读取衰减报告失败: {e}，使用默认调仓周期 {args.rebalance_freq}d")
    
    if decay_half_life is not None:
        # 用户未手动指定时才覆盖（检测是否为默认值）
        if args.rebalance_freq == 20:  # 默认值
            args.rebalance_freq = decay_half_life
            print(f"[信息] 调仓周期自动设置为 {args.rebalance_freq} 天")
        else:
            print(f"[信息] 用户手动指定调仓周期 {args.rebalance_freq}d，忽略衰减报告的 {decay_half_life}d")
        # forward_days 也联动
        if args.forward_days == 20:  # 默认值
            args.forward_days = decay_half_life
            print(f"[信息] IC 前瞻天数自动设置为 {args.forward_days} 天")
    
    # 加载
    print(f"[信息] 加载因子值: {args.factor}")
    factor_matrix, factor_col = load_matrix(args.factor)
    factor_name = args.factor_name or factor_col
    print(f"[信息] 因子矩阵: {factor_matrix.shape[0]} 日 × {factor_matrix.shape[1]} 股")
    
    print(f"[信息] 加载收益率: {args.returns}")
    return_matrix, ret_col = load_matrix(args.returns)
    if args.is_price:
        print(f"[信息] 价格 → 日收益率...")
        return_matrix = compute_returns_from_prices(return_matrix)
    print(f"[信息] 收益矩阵: {return_matrix.shape[0]} 日 × {return_matrix.shape[1]} 股")
    
    # 对齐
    common_dates = sorted(factor_matrix.index.intersection(return_matrix.index))
    common_stocks = sorted(factor_matrix.columns.intersection(return_matrix.columns))
    print(f"[信息] 重叠: {len(common_dates)} 日, {len(common_stocks)} 股")
    
    if len(common_dates) < 10 or len(common_stocks) < 5:
        print("[错误] 数据不足", file=sys.stderr)
        sys.exit(1)
    
    factor_matrix = factor_matrix.loc[common_dates, common_stocks]
    return_matrix = return_matrix.loc[common_dates, common_stocks]
    
    # 涨跌停/停牌过滤
    limit_filter = None
    if args.limit_filter:
        print(f"[信息] 加载涨跌停过滤: {args.limit_filter}")
        lf_df = pd.read_csv(args.limit_filter, encoding="utf-8")
        lf_df["date"] = pd.to_datetime(lf_df["date"])
        if "tradable" in lf_df.columns:
            lf_df["tradable"] = lf_df["tradable"].astype(bool)
            limit_filter = lf_df.pivot_table(
                index="date", columns="stock_code", values="tradable",
                fill_value=True
            )
        else:
            print("[警告] 过滤文件缺少 'tradable' 列，跳过", file=sys.stderr)
    
    # 检测静态因子
    static = args.static_factor or is_static_factor(factor_matrix)
    if static:
        print(f"[信息] 静态因子 → IC 使用不重叠窗口（每 {args.forward_days} 天）")
    
    # IC
    print(f"[信息] 计算 IC (forward={args.forward_days}d)...")
    if static:
        factor_avg = factor_matrix.mean(axis=0)
        ic_series = compute_ic_static(factor_avg, return_matrix,
                                       forward_days=args.forward_days, method="pearson")
        rank_ic_series = compute_ic_static(factor_avg, return_matrix,
                                            forward_days=args.forward_days, method="spearman")
    else:
        ic_series = compute_ic_dynamic(factor_matrix, return_matrix,
                                        forward_days=args.forward_days, method="pearson")
        rank_ic_series = compute_ic_dynamic(factor_matrix, return_matrix,
                                             forward_days=args.forward_days, method="spearman")
    
    # 分层回测
    cost_desc = f", 成本{args.cost*100:.2f}%" if args.cost > 0 else ""
    filter_desc = ", 涨跌停过滤" if limit_filter is not None else ""
    print(f"[信息] 分层: {args.n_groups}组, 频率{args.rebalance_freq}天{cost_desc}{filter_desc}...")
    group_returns, turnovers, holdings_info = compute_group_returns(
        factor_matrix, return_matrix, args.n_groups,
        args.rebalance_freq, args.cost, limit_filter
    )
    
    # 指标
    print(f"[信息] 计算指标...")
    metrics = compute_metrics(
        group_returns, ic_series, rank_ic_series, turnovers, args.n_groups,
        holdings_info=holdings_info
    )
    
    # 报告
    report = {
        "factor_name": factor_name,
        "static_factor": static,
        "period": (f"{common_dates[0].strftime('%Y-%m-%d')} ~ "
                   f"{common_dates[-1].strftime('%Y-%m-%d')}"),
        "n_dates": len(common_dates),
        "n_stocks": len(common_stocks),
        "n_groups": args.n_groups,
        "rebalance_freq": args.rebalance_freq,
        "forward_days": args.forward_days,
        "cost_per_trade": args.cost,
        "metrics": metrics,
    }
    
    # NaN → null, numpy types → Python native
    def nan_to_none(obj):
        if isinstance(obj, (np.bool_, )):
            return bool(obj)
        if isinstance(obj, (np.integer, )):
            return int(obj)
        if isinstance(obj, (np.floating, )):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
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
    print(f"[信息] 报告已保存: {args.output_report}")
    
    # 中间数据
    if args.output_dir:
        print(f"[信息] 保存中间数据: {args.output_dir}")
        save_backtest_data(group_returns, ic_series, rank_ic_series, args.output_dir)
    
    # 摘要
    print(f"\n{'='*60}")
    print(f"  单因子回测 v2: {factor_name}")
    if static:
        print(f"  [静态因子] IC 基于不重叠 {args.forward_days}d 窗口")
    print(f"{'='*60}")
    print(f"  区间:   {report['period']}")
    print(f"  股票:   {len(common_stocks)}")
    print(f"  分组:   {args.n_groups}  调仓: {args.rebalance_freq}天")
    if args.cost > 0:
        print(f"  成本:   单边 {args.cost*100:.2f}%")
    print(f"{'─'*60}")
    
    ic_sig = "✓" if metrics.get("ic_significant_5pct") else "✗"
    print(f"  IC 均值:     {metrics.get('ic_mean', np.nan):>8.4f}  "
          f"(t={metrics.get('ic_t_stat', np.nan):.2f}, {ic_sig} @5%)")
    print(f"  IC 标准差:   {metrics.get('ic_std', np.nan):>8.4f}")
    print(f"  IC > 0:      {metrics.get('ic_positive_pct', np.nan):>8.1%}  "
          f"(N={metrics.get('ic_count', 0)})")
    print(f"  Rank IC:     {metrics.get('rank_ic_mean', np.nan):>8.4f}")
    print(f"  IR:          {metrics.get('ir', np.nan):>8.4f}")
    print(f"{'─'*60}")
    
    ls_tr = metrics.get('long_short_total_return', np.nan)
    ls_ar = metrics.get('long_short_ann_return', np.nan)
    ls_sh = metrics.get('long_short_sharpe', np.nan)
    ls_md = metrics.get('long_short_mdd', np.nan)
    print(f"  多空总收益:  {ls_tr:>8.2%}")
    print(f"  多空年化:    {ls_ar:>8.2%}")
    print(f"  多空 Sharpe: {ls_sh:>8.4f}")
    print(f"  多空 MDD:    {ls_md:>8.2%}")
    print(f"{'─'*60}")
    
    print(f"  换手率均值:  {metrics.get('turnover_mean', np.nan):>8.2%}")
    print(f"  单调性:      {metrics.get('monotonicity', np.nan):>8.4f}")
    
    # 持仓诊断
    hi = metrics.get("holdings_info", {})
    if hi:
        g1 = hi.get("group_1", {})
        gn = hi.get(f"group_{args.n_groups}", {})
        print(f"  G1 持仓:     均{g1.get('mean',0):.0f}, "
              f"范围[{g1.get('min',0)}-{g1.get('max',0)}]")
        print(f"  G{args.n_groups} 持仓:     均{gn.get('mean',0):.0f}, "
              f"范围[{gn.get('min',0)}-{gn.get('max',0)}]")
    warn = metrics.get("holdings_warning")
    if warn:
        print(f"  ⚠ {warn}")
    print(f"{'─'*60}")
    
    print(f"  分层年化收益 (低→高):")
    for i, ret in enumerate(metrics.get("group_returns_annualized", []), 1):
        bar = "█" * max(int((ret or 0) * 100), 0)
        ret_str = f"{ret:>8.2%}" if ret is not None and not np.isnan(ret) else "    N/A"
        print(f"    G{i}: {ret_str}  {bar}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()