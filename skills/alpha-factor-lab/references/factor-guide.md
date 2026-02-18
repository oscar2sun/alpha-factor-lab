# 因子定义与公式编写指南

## 一、常用因子分类

### 1. 价值因子 (Value)
衡量股票价格相对于基本面的便宜程度。

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `ep` | `net_profit / market_cap` | 盈利收益率（PE 的倒数） |
| `bp` | `book_value / market_cap` | 账面市值比（PB 的倒数） |
| `sp` | `revenue / market_cap` | 营收市值比（PS 的倒数） |
| `cfp` | `operating_cash_flow / market_cap` | 现金流市值比 |
| `dp` | `dividend / market_cap` | 股息率 |

### 2. 动量因子 (Momentum)
衡量股票过去一段时间的价格趋势。

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `mom_1m` | `close / close_lag_20 - 1` | 1个月动量 |
| `mom_3m` | `close / close_lag_60 - 1` | 3个月动量 |
| `mom_6m` | `close / close_lag_120 - 1` | 6个月动量 |
| `mom_12m` | `close_lag_20 / close_lag_250 - 1` | 12个月动量（跳过最近1月） |
| `reversal` | `-(close / close_lag_5 - 1)` | 5日反转 |

### 3. 质量因子 (Quality)
衡量公司盈利质量和财务健康度。

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `roe` | `net_profit / equity` | 净资产收益率 |
| `roa` | `net_profit / total_assets` | 总资产收益率 |
| `gross_margin` | `gross_profit / revenue` | 毛利率 |
| `accruals` | `(net_profit - operating_cash_flow) / total_assets` | 应计项目比率（低优） |
| `debt_ratio` | `total_debt / total_assets` | 资产负债率（低优） |

### 4. 成长因子 (Growth)
衡量公司成长性。

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `revenue_growth` | `revenue / revenue_prev - 1` | 营收增长率 |
| `profit_growth` | `net_profit / net_profit_prev - 1` | 净利润增长率 |
| `roe_growth` | `roe / roe_prev - 1` | ROE 变化率 |
| `rd_intensity` | `rd_expense / revenue` | 研发强度 |

### 5. 波动率因子 (Volatility)
衡量股票价格的波动程度（通常低波动为优）。

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `volatility_20` | `returns_std_20` | 20日收益率标准差 |
| `volatility_60` | `returns_std_60` | 60日收益率标准差 |
| `idio_vol` | `residual_std_60` | 特质波动率（剥离市场因素） |
| `downside_vol` | `downside_std_60` | 下行波动率 |

### 6. 流动性因子 (Liquidity)
衡量股票的流动性（通常低流动性有溢价）。

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `turnover_20` | `volume_mean_20 / float_shares` | 20日平均换手率 |
| `amihud` | `abs_return_mean_20 / amount_mean_20` | Amihud 非流动性指标 |
| `volume_ratio` | `volume_mean_5 / volume_mean_20` | 量比 |

### 7. 技术因子 (Technical)
基于价量关系的技术指标。

| 因子名 | 公式 | 说明 |
|--------|------|------|
| `rsi_14` | `rsi(close, 14)` | 14日 RSI |
| `macd_signal` | `ema(close, 12) - ema(close, 26)` | MACD 信号 |
| `price_volume_corr` | `corr(close, volume, 20)` | 价量相关性 |

## 二、因子公式编写规范

### 可用变量名
在 `factor_calculator.py` 中，以下变量名可直接在公式中使用：

**价格数据（来自 K 线）：**
- `open`, `high`, `low`, `close` — 开高低收
- `volume` — 成交量
- `amount` — 成交额
- `close_lag_N` — N 天前的收盘价（N = 5, 10, 20, 60, 120, 250）
- `returns` — 日收益率 `close / close.shift(1) - 1`

**财务数据（来自财报）：**
- `revenue` — 营业收入
- `net_profit` — 归母净利润
- `gross_profit` — 毛利润
- `operating_cash_flow` — 经营性现金流
- `total_assets` — 总资产
- `total_debt` — 总负债
- `equity` — 净资产（股东权益）
- `book_value` — 每股净资产
- `market_cap` — 总市值
- `float_shares` — 流通股本
- `earnings_per_share` — 每股收益
- `dividend` — 每股股利
- `rd_expense` — 研发费用

**衍生统计量：**
- `returns_std_N` — N 日收益率标准差
- `volume_mean_N` — N 日平均成交量
- `amount_mean_N` — N 日平均成交额
- `abs_return_mean_N` — N 日平均绝对收益率

### 公式编写规则
1. 使用标准 Python 数学表达式
2. 支持 `numpy` 函数：`np.log()`, `np.abs()`, `np.sign()`, `np.where()` 等
3. 支持 `pandas` 方法链：`.rolling()`, `.shift()`, `.rank()`, `.pct_change()` 等
4. 分母为零时自动替换为 NaN
5. 因子值在横截面上自动标准化（z-score）

### 公式示例
```python
# 盈利收益率
"net_profit / market_cap"

# 动量（3个月）
"close / close_lag_60 - 1"

# ROE 改善
"roe - roe_prev"

# 价量背离
"-1 * corr(close, volume, 20)"

# 复合因子
"0.5 * (net_profit / market_cap) + 0.3 * (revenue / revenue_prev - 1) + 0.2 * (-returns_std_20)"
```

## 三、因子评价标准

### 核心指标

#### IC (Information Coefficient)
- **定义**：因子值与下一期收益率的截面相关系数（Pearson 或 Spearman）
- **计算**：每期计算一个 IC，取时序均值
- **标准**：|IC Mean| > 0.03 即有预测能力，> 0.05 为优秀

#### Rank IC
- **定义**：因子排名与收益排名的 Spearman 秩相关系数
- **优势**：对异常值更鲁棒
- **通常比 Pearson IC 更稳定**

#### IR (Information Ratio)
- **定义**：IC Mean / IC Std
- **含义**：因子预测能力的稳定性
- **标准**：> 0.3 为可用，> 0.5 为优秀

#### 分层单调性
- **定义**：按因子值排序分成 N 组，各组收益是否单调递增/递减
- **最关键的直觉检验**：如果不单调，因子逻辑可能有问题

#### 多空收益 (Long-Short Return)
- **定义**：做多因子值最高组，做空最低组的收益
- **衡量因子可获取的超额收益**

#### 换手率 (Turnover)
- **定义**：相邻两期组合成分股的变化比例
- **影响实际可执行性**：换手率太高（> 0.5）实盘难以盈利

#### 半衰期 (Half-life)
- **定义**：因子自相关系数衰减到 0.5 所需的时间
- **含义**：因子信息的持续性
- **标准**：> 20 天为好，< 10 天说明信号衰减太快

### 综合评估矩阵
| 维度 | 优秀 | 良好 | 一般 | 差 |
|------|------|------|------|----|
| |IC Mean| | > 0.05 | 0.03~0.05 | 0.02~0.03 | < 0.02 |
| IR | > 0.5 | 0.3~0.5 | 0.2~0.3 | < 0.2 |
| 多空 Sharpe | > 1.5 | 1.0~1.5 | 0.5~1.0 | < 0.5 |
| 分层单调性 | 完全单调 | 基本单调 | 部分单调 | 不单调 |
| Half-life | > 20天 | 10~20天 | 5~10天 | < 5天 |
| Turnover | < 0.2 | 0.2~0.35 | 0.35~0.5 | > 0.5 |

## 四、常见陷阱

### 1. 前视偏差 (Look-ahead Bias)
- **问题**：使用了未来才能获得的数据
- **常见场景**：
  - 财报数据使用报告期而非发布日期
  - 用当天收盘价计算当天的因子并选股
- **解决**：所有数据至少滞后 1 天；财报数据滞后到发布日之后

### 2. 存活偏差 (Survivorship Bias)
- **问题**：只用当前存续的股票回测，忽略了已退市的股票
- **影响**：高估因子收益（退市的通常是差的）
- **解决**：使用历史成分股数据，包含已退市股票

### 3. 过拟合 (Overfitting)
- **问题**：因子在样本内表现好，样本外失效
- **常见信号**：
  - IC 非常高（> 0.10）但不稳定
  - 参数敏感性很高
  - 逻辑说不通但数字好看
- **解决**：样本内/外分割检验；因子逻辑必须有经济学解释

### 4. 行业暴露 (Industry Exposure)
- **问题**：因子实际上只是行业的代理变量
- **例子**：高 PB 因子实际上是做空银行做多科技
- **解决**：做行业中性化处理（行业内排名或回归剔除）

### 5. 市值暴露 (Size Exposure)
- **问题**：很多因子与市值高度相关
- **解决**：对市值做中性化，或控制市值分组后再测试

### 6. 交易成本忽略
- **问题**：回测不考虑交易成本，高换手率因子实盘无法盈利
- **解决**：在回测中扣除合理的交易成本（单边 0.1%~0.3%）

### 7. 数据质量
- **问题**：异常值、缺失值、数据错误
- **解决**：
  - 极端值处理：MAD 或 Winsorize（1%/99% 分位截断）
  - 缺失值：同行业/同市值中位数填充，或直接剔除
  - 交叉验证不同数据源

## 五、MCP 数据工具与因子变量映射

| 因子变量 | MCP 工具 | 字段 |
|----------|----------|------|
| `close`, `open`, `high`, `low`, `volume`, `amount` | `fintool-quote.get_kline` | K 线数据 |
| `revenue`, `net_profit`, `gross_profit` | `fintool-company.get_income_statement` | 利润表 |
| `total_assets`, `total_debt`, `equity` | `fintool-company.get_balance_sheet` | 资产负债表 |
| `operating_cash_flow` | `fintool-company.get_cash_flow` | 现金流量表 |
| `market_cap`, `pe`, `pb` | `fintool-company.get_valuation_metrics_daily` | 估值指标 |
| `roe`, `roa`, `eps` | `fintool-company.get_financial_indicators` | 财务衍生指标 |
| `rd_expense` | `fintool-company.get_research_development_expense` | 研发投入 |
