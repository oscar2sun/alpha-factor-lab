#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算引擎 v2 (Factor Calculator)

功能：
  - 读取股票池数据（CSV/JSON）
  - 根据因子公式（Python 表达式）计算因子值
  - AST 白名单安全公式执行（禁止 import/exec/eval）
  - 按需计算衍生字段（只计算公式中引用的变量）
  - 横截面标准化（z-score）
  - 行业中性化（虚拟变量 OLS 回归取残差）
  - 多变量中性化（市值 + 行业联合回归）
  - Winsorize: MAD 方法 或 百分位截断
  - 对数变换 / Rank 变换

v2 改进（相比 v1）：
  - eval() → AST 白名单安全执行
  - 预计算全部衍生字段 → 按需计算（公式引用了什么才算什么）
  - 新增 --industry-col 行业中性化
  - 新增 --log-transform / --rank-transform
  - --neutralize 支持多变量（逗号分隔）
  - Winsorize 默认改为 MAD 方法

用法：
  python3 factor_calculator.py \\
    --formula "net_profit / market_cap" \\
    --data stock_data.csv \\
    --output factor_values.csv \\
    [--neutralize market_cap] \\
    [--industry-col industry] \\
    [--winsorize 3.0] \\
    [--winsorize-method mad|percentile] \\
    [--log-transform] \\
    [--rank-transform] \\
    [--no-zscore]
"""

import argparse
import ast
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# AST 白名单安全公式执行
# ============================================================

# 允许的 AST 节点类型
_ALLOWED_NODES = {
    ast.Expression, ast.Module,
    # 字面量
    ast.Constant, ast.Num, ast.Str,
    # 变量
    ast.Name, ast.Load,
    # 运算
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    # 函数调用
    ast.Call, ast.keyword,
    # 属性访问（如 x.rolling()）
    ast.Attribute,
    # 下标（如 x[mask]）
    ast.Subscript, ast.Index, ast.Slice,
    # 条件表达式
    ast.IfExp,
    # Tuple（用于多返回值等）
    ast.Tuple,
}

# 禁止的函数/属性名
_FORBIDDEN_NAMES = {
    "__import__", "exec", "eval", "compile", "globals", "locals",
    "__builtins__", "__class__", "__subclasses__", "__bases__",
    "getattr", "setattr", "delattr", "open", "input",
    "breakpoint", "exit", "quit",
}


def _validate_ast(node: ast.AST) -> None:
    """递归验证 AST 节点，拒绝不在白名单中的操作。"""
    if type(node) not in _ALLOWED_NODES:
        raise ValueError(
            f"公式中不允许使用 {type(node).__name__} "
            f"(line {getattr(node, 'lineno', '?')})"
        )
    
    # 检查名称
    if isinstance(node, ast.Name):
        if node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"公式中禁止使用名称 '{node.id}'")
    
    if isinstance(node, ast.Attribute):
        if node.attr in _FORBIDDEN_NAMES:
            raise ValueError(f"公式中禁止访问属性 '{node.attr}'")
    
    # 递归检查子节点
    for child in ast.iter_child_nodes(node):
        _validate_ast(child)


def _extract_names(formula: str) -> set:
    """从公式中提取引用的变量名（用于按需计算衍生字段）。"""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return set()
    
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def safe_eval_formula(df: pd.DataFrame, formula: str) -> pd.Series:
    """
    安全地计算因子公式。

    使用 AST 白名单验证公式安全性，然后在受限命名空间中执行。
    """
    # Step 1: AST 安全检查
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        print(f"[错误] 公式语法错误: {formula}", file=sys.stderr)
        print(f"[错误] 详情: {e}", file=sys.stderr)
        sys.exit(1)
    
    _validate_ast(tree)
    
    # Step 2: 构建安全命名空间
    local_ns = {
        col: df[col]
        for col in df.columns
        if col not in ("date", "stock_code")
    }
    local_ns["np"] = np
    local_ns["pd"] = pd
    local_ns["log"] = np.log
    local_ns["log1p"] = np.log1p
    local_ns["abs"] = np.abs
    local_ns["sign"] = np.sign
    local_ns["where"] = np.where
    local_ns["rank"] = lambda x: x.rank(pct=True)
    local_ns["sqrt"] = np.sqrt
    local_ns["exp"] = np.exp
    local_ns["max"] = np.maximum
    local_ns["min"] = np.minimum
    
    def corr(x, y, window):
        return x.rolling(window, min_periods=max(window // 2, 1)).corr(y)
    
    def ema(x, span):
        return x.ewm(span=span, adjust=False).mean()
    
    def rsi(x, period):
        delta = x.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)
    
    def ts_rank(x, window):
        return x.rolling(window, min_periods=max(window // 2, 1)).rank(pct=True)
    
    def ts_mean(x, window):
        return x.rolling(window, min_periods=max(window // 2, 1)).mean()
    
    def ts_std(x, window):
        return x.rolling(window, min_periods=max(window // 2, 1)).std()
    
    def ts_max(x, window):
        return x.rolling(window, min_periods=max(window // 2, 1)).max()
    
    def ts_min(x, window):
        return x.rolling(window, min_periods=max(window // 2, 1)).min()
    
    local_ns.update({
        "corr": corr, "ema": ema, "rsi": rsi,
        "ts_rank": ts_rank, "ts_mean": ts_mean, "ts_std": ts_std,
        "ts_max": ts_max, "ts_min": ts_min,
    })
    
    # Step 3: 执行
    try:
        code = compile(tree, "<formula>", "eval")
        with np.errstate(divide="ignore", invalid="ignore"):
            result = eval(code, {"__builtins__": {}}, local_ns)
        
        if isinstance(result, (int, float)):
            result = pd.Series(result, index=df.index)
        
        result = result.replace([np.inf, -np.inf], np.nan)
        return result
    
    except Exception as e:
        print(f"[错误] 公式计算失败: {formula}", file=sys.stderr)
        print(f"[错误] 详情: {e}", file=sys.stderr)
        available = [c for c in df.columns if c not in ("date", "stock_code")]
        print(f"[信息] 可用变量: {available}", file=sys.stderr)
        sys.exit(1)


# ============================================================
# 数据加载
# ============================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    加载数据文件，支持 CSV 和 JSON。

    期望的数据格式：
    - CSV: 列包含 date, stock_code, 以及各数据字段
    - JSON: 列表/嵌套字典
    """
    path = Path(filepath)
    if not path.exists():
        print(f"[错误] 数据文件不存在: {filepath}", file=sys.stderr)
        sys.exit(1)
    
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            records = []
            for stock, dates in raw.items():
                if isinstance(dates, dict):
                    for date, fields in dates.items():
                        record = {"stock_code": stock, "date": date}
                        if isinstance(fields, dict):
                            record.update(fields)
                        records.append(record)
                else:
                    records.append({"stock_code": stock, **dates})
            df = pd.DataFrame(records)
        else:
            print(f"[错误] 无法解析 JSON 格式", file=sys.stderr)
            sys.exit(1)
    else:
        df = pd.read_csv(path, encoding="utf-8")
    
    if "date" not in df.columns or "stock_code" not in df.columns:
        print(f"[错误] 数据必须包含 'date' 和 'stock_code' 列", file=sys.stderr)
        print(f"[信息] 当前列: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "stock_code"]).reset_index(drop=True)
    
    return df


# ============================================================
# 按需衍生字段
# ============================================================

def compute_derived_fields(df: pd.DataFrame, needed_names: set) -> pd.DataFrame:
    """
    按需计算衍生字段。只计算公式中实际引用的变量。
    """
    df = df.copy()
    grouped = df.groupby("stock_code")
    
    # 日收益率
    if "returns" in needed_names and "returns" not in df.columns:
        if "close" in df.columns:
            df["returns"] = grouped["close"].pct_change()
    
    # 滞后收盘价: close_lag_N
    lag_pattern = {5, 10, 20, 60, 120, 250}
    for lag in lag_pattern:
        col = f"close_lag_{lag}"
        if col in needed_names and col not in df.columns:
            if "close" in df.columns:
                df[col] = grouped["close"].shift(lag)
    
    # 滚动统计量: returns_std_N
    for window in [20, 60]:
        col = f"returns_std_{window}"
        if col in needed_names and col not in df.columns:
            if "returns" not in df.columns and "close" in df.columns:
                df["returns"] = grouped["close"].pct_change()
            if "returns" in df.columns:
                df[col] = grouped["returns"].transform(
                    lambda x: x.rolling(window, min_periods=max(window // 2, 1)).std()
                )
    
    # volume_mean_N
    for window in [5, 20]:
        col = f"volume_mean_{window}"
        if col in needed_names and col not in df.columns:
            if "volume" in df.columns:
                df[col] = grouped["volume"].transform(
                    lambda x: x.rolling(window, min_periods=max(window // 2, 1)).mean()
                )
    
    # amount_mean_N
    for window in [20]:
        col = f"amount_mean_{window}"
        if col in needed_names and col not in df.columns:
            if "amount" in df.columns:
                df[col] = grouped["amount"].transform(
                    lambda x: x.rolling(window, min_periods=max(window // 2, 1)).mean()
                )
    
    # abs_return_mean_N
    for window in [20]:
        col = f"abs_return_mean_{window}"
        if col in needed_names and col not in df.columns:
            if "returns" not in df.columns and "close" in df.columns:
                df["returns"] = grouped["close"].pct_change()
            if "returns" in df.columns:
                df[col] = grouped["returns"].transform(
                    lambda x: x.abs().rolling(window, min_periods=max(window // 2, 1)).mean()
                )
    
    # log_market_cap（常用的市值中性化变量）
    if "log_market_cap" in needed_names and "log_market_cap" not in df.columns:
        if "market_cap" in df.columns:
            df["log_market_cap"] = np.log(df["market_cap"].replace(0, np.nan))
    
    return df


# ============================================================
# Winsorize 方法
# ============================================================

def winsorize_percentile(series: pd.Series, limits: float = 0.01) -> pd.Series:
    """百分位截断法。"""
    lower = series.quantile(limits)
    upper = series.quantile(1 - limits)
    return series.clip(lower, upper)


def winsorize_mad(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
    """
    MAD (Median Absolute Deviation) 方法。
    
    超过 median ± n_mad * 1.4826 * MAD 的值截断。
    1.4826 是使 MAD 与正态分布标准差一致的常数。
    """
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return series
    scaled_mad = 1.4826 * mad  # 使 MAD 与 σ 一致
    lower = med - n_mad * scaled_mad
    upper = med + n_mad * scaled_mad
    return series.clip(lower, upper)


# ============================================================
# 中性化（行业 + 多变量）
# ============================================================

def _ols_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    用 numpy 手写 OLS 回归，返回残差。
    y: (n,) 因子值
    X: (n, k) 自变量矩阵（含截距列）
    """
    # 过滤 NaN
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < X.shape[1] + 2:
        return y  # 样本不足，返回原值
    
    y_clean = y[mask]
    X_clean = X[mask]
    
    # OLS: beta = (X'X)^{-1} X'y
    try:
        XtX = X_clean.T @ X_clean
        Xty = X_clean.T @ y_clean
        # 加微小正则化防止奇异
        beta = np.linalg.solve(XtX + 1e-10 * np.eye(XtX.shape[0]), Xty)
        residual = np.full_like(y, np.nan)
        residual[mask] = y_clean - X_clean @ beta
        return residual
    except np.linalg.LinAlgError:
        return y


def neutralize_cross_section(
    df: pd.DataFrame,
    factor_col: str,
    neutralize_cols: list = None,
    industry_col: str = None,
) -> pd.Series:
    """
    横截面中性化：对因子值做行业+市值联合回归，取残差。
    
    - neutralize_cols: 连续变量（如 market_cap, log_market_cap）
    - industry_col: 行业分类列（生成 dummy 变量）
    
    在每个日期的截面上独立做回归。
    """
    result = pd.Series(np.nan, index=df.index)
    
    for date, group in df.groupby("date"):
        y = group[factor_col].values.astype(float)
        valid = np.isfinite(y)
        
        if valid.sum() < 5:
            continue
        
        # 构建 X 矩阵
        x_parts = []
        
        # 截距
        x_parts.append(np.ones(len(group)))
        
        # 连续变量
        if neutralize_cols:
            for col in neutralize_cols:
                if col in group.columns:
                    x_parts.append(group[col].values.astype(float))
        
        # 行业哑变量
        if industry_col and industry_col in group.columns:
            industries = group[industry_col].values
            unique_ind = sorted(set(industries[pd.notna(industries)]))
            if len(unique_ind) > 1:
                # N-1 个哑变量（去掉第一个作为基准）
                for ind in unique_ind[1:]:
                    dummy = (industries == ind).astype(float)
                    x_parts.append(dummy)
        
        X = np.column_stack(x_parts)
        residual = _ols_residual(y, X)
        result.loc[group.index] = residual
    
    return result


def zscore_cross_section(df: pd.DataFrame, col: str) -> pd.Series:
    """纯横截面 z-score 标准化。"""
    result = pd.Series(np.nan, index=df.index)
    
    for date, group in df.groupby("date"):
        vals = group[col] if isinstance(col, str) else col.loc[group.index]
        valid = vals.notna()
        if valid.sum() < 3:
            continue
        v = vals[valid]
        std = v.std()
        if std > 0:
            result.loc[group.index[valid]] = (v - v.mean()) / std
    
    return result


# ============================================================
# 变换
# ============================================================

def log_transform(series: pd.Series) -> pd.Series:
    """sign(x) * log(1 + |x|) 变换，处理右偏分布。"""
    return np.sign(series) * np.log1p(series.abs())


def rank_transform(df: pd.DataFrame, col: str) -> pd.Series:
    """截面百分位排名变换 (0~1)。"""
    result = pd.Series(np.nan, index=df.index)
    for date, group in df.groupby("date"):
        vals = group[col]
        valid = vals.notna()
        if valid.sum() < 3:
            continue
        result.loc[group.index[valid]] = vals[valid].rank(pct=True)
    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="因子计算引擎 v2")
    parser.add_argument("--formula", required=True,
                        help="因子公式（Python表达式）")
    parser.add_argument("--data", required=True,
                        help="输入数据文件路径（CSV/JSON）")
    parser.add_argument("--output", required=True,
                        help="输出因子值 CSV 路径")
    parser.add_argument("--neutralize", default=None,
                        help="中性化连续变量（逗号分隔，如 market_cap,log_market_cap）")
    parser.add_argument("--industry-col", default=None,
                        help="行业分类列名（用于行业中性化，如 industry）")
    parser.add_argument("--winsorize", type=float, default=3.0,
                        help="Winsorize 参数: MAD倍数(默认3.0) 或 百分位(如0.01)")
    parser.add_argument("--winsorize-method", choices=["mad", "percentile"],
                        default="mad", help="Winsorize 方法（默认 MAD）")
    parser.add_argument("--log-transform", action="store_true",
                        help="对原始因子值做 sign(x)*log(1+|x|) 变换")
    parser.add_argument("--rank-transform", action="store_true",
                        help="将因子值转为截面百分位排名 (0~1)")
    parser.add_argument("--no-zscore", action="store_true",
                        help="跳过 z-score 标准化")
    parser.add_argument("--factor-name", default="factor",
                        help="因子名称（用于输出列名）")
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"[信息] 加载数据: {args.data}")
    df = load_data(args.data)
    print(f"[信息] 数据规模: {len(df)} 行, {df['stock_code'].nunique()} 只股票, "
          f"{df['date'].nunique()} 个交易日")
    
    # 提取公式引用的变量名 → 按需计算衍生字段
    needed = _extract_names(args.formula)
    # 加上中性化变量
    neutralize_cols = []
    if args.neutralize:
        neutralize_cols = [c.strip() for c in args.neutralize.split(",") if c.strip()]
        needed.update(neutralize_cols)
    
    print(f"[信息] 公式引用变量: {needed & set(df.columns) | (needed - set(df.columns))}")
    print(f"[信息] 按需计算衍生字段...")
    df = compute_derived_fields(df, needed)
    
    # 计算因子
    print(f"[信息] 计算因子: {args.formula}")
    df["_raw_factor"] = safe_eval_formula(df, args.formula)
    
    valid_count = df["_raw_factor"].notna().sum()
    total_count = len(df)
    print(f"[信息] 原始因子有效值: {valid_count}/{total_count} "
          f"({valid_count / total_count * 100:.1f}%)")
    
    if valid_count == 0:
        print("[错误] 因子值全部为 NaN，请检查公式和数据", file=sys.stderr)
        sys.exit(1)
    
    # 对数变换
    if args.log_transform:
        print(f"[信息] 对数变换: sign(x) * log(1 + |x|)")
        df["_raw_factor"] = log_transform(df["_raw_factor"])
    
    # Winsorize
    if args.winsorize > 0:
        method = args.winsorize_method
        param = args.winsorize
        print(f"[信息] Winsorize ({method}, param={param})")
        if method == "mad":
            df["_raw_factor"] = df.groupby("date")["_raw_factor"].transform(
                lambda x: winsorize_mad(x, param)
            )
        else:
            df["_raw_factor"] = df.groupby("date")["_raw_factor"].transform(
                lambda x: winsorize_percentile(x, param)
            )
    
    # 中性化
    need_neutralize = bool(neutralize_cols) or bool(args.industry_col)
    if need_neutralize:
        desc = []
        if neutralize_cols:
            desc.append(f"连续变量: {neutralize_cols}")
        if args.industry_col:
            desc.append(f"行业: {args.industry_col}")
        print(f"[信息] 中性化回归 ({', '.join(desc)})")
        
        df["_neutralized"] = neutralize_cross_section(
            df, "_raw_factor",
            neutralize_cols=neutralize_cols if neutralize_cols else None,
            industry_col=args.industry_col,
        )
        
        # 二次 Winsorize：中性化残差可能产生新的异常值
        if args.winsorize > 0:
            method = args.winsorize_method
            param = args.winsorize
            print(f"[信息] 二次 Winsorize（中性化残差, {method}, param={param}）")
            if method == "mad":
                df["_neutralized"] = df.groupby("date")["_neutralized"].transform(
                    lambda x: winsorize_mad(x, param)
                )
            else:
                df["_neutralized"] = df.groupby("date")["_neutralized"].transform(
                    lambda x: winsorize_percentile(x, param)
                )
        
        factor_source = "_neutralized"
    else:
        factor_source = "_raw_factor"
    
    # Rank 变换
    if args.rank_transform:
        print(f"[信息] Rank 变换: 截面百分位排名 (0~1)")
        df[args.factor_name] = rank_transform(df, factor_source)
    elif not args.no_zscore:
        print(f"[信息] 横截面 Z-score 标准化")
        df[args.factor_name] = zscore_cross_section(df, factor_source)
    else:
        df[args.factor_name] = df[factor_source]
    
    # 输出
    output_cols = ["date", "stock_code", args.factor_name]
    output_df = df[output_cols].copy()
    output_df["date"] = output_df["date"].dt.strftime("%Y-%m-%d")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    
    # 统计摘要
    fv = output_df[args.factor_name]
    print(f"\n[结果] 因子值已保存到: {args.output}")
    print(f"[统计] 有效值: {fv.notna().sum()}")
    print(f"[统计] 均值:   {fv.mean():.4f}")
    print(f"[统计] 标准差: {fv.std():.4f}")
    print(f"[统计] 最小值: {fv.min():.4f}")
    print(f"[统计] 最大值: {fv.max():.4f}")
    print(f"[统计] 中位数: {fv.median():.4f}")
    print(f"[统计] 偏度:   {fv.skew():.4f}")
    print(f"[统计] 峰度:   {fv.kurtosis():.4f}")


if __name__ == "__main__":
    main()
