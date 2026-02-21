---
name: us-market
description: 美股市场信息查询。当用户需要查询美股股价、行情、财报、指数、公司基本面、分析师评级、历史K线、分红、ETF等信息时使用。支持按 ticker（如 AAPL、TSLA）或公司名查询。覆盖 NYSE、NASDAQ、AMEX 全市场。内置主/备数据源自动切换机制。
---

# 美股市场信息查询

## 数据源架构

采用主/备数据源机制，当主数据源不可用时自动切换。

### 数据源清单

| 数据源 | 类型 | 免费额度 | 擅长领域 |
|--------|------|----------|----------|
| **yfinance** | Python库 | 无限（有速率限制） | 股价、财报、指数、分红、公司信息 |
| **Alpha Vantage** | REST API | 25次/天 | 股价历史（20年+）、技术指标、外汇 |
| **FMP** | REST API | 250次/天 | 财报、估值、筛选器、SEC文件 |
| **Finnhub** | REST API | 60次/分 | 实时行情、新闻、分析师评级、内部交易 |
| **finmap MCP** | MCP | 无限 | 市场板块热力图、板块分布 |

### 查询分类与数据源映射

| 查询类型 | 主数据源 | 备用数据源 | 说明 |
|----------|----------|------------|------|
| 实时行情/股价 | yfinance | Finnhub → Alpha Vantage | 当前价、涨跌幅、成交量 |
| 历史K线 | yfinance | Alpha Vantage | 日/周/月线，最长20年+ |
| 财务报表 | yfinance | FMP | 利润表、资产负债表、现金流 |
| 公司概况 | yfinance | FMP | 市值、行业、描述、员工数 |
| 分析师评级 | yfinance | Finnhub | 目标价、买卖评级 |
| 新闻资讯 | Finnhub | web_search | 公司新闻、行业新闻 |
| 指数行情 | yfinance | Alpha Vantage | S&P500、NASDAQ、道琼斯 |
| 分红记录 | yfinance | FMP | 分红历史、股息率 |
| 内部交易 | Finnhub | — | 高管买卖记录 |
| 板块热力图 | finmap MCP | — | 美股板块分布 |
| 技术指标 | Alpha Vantage | yfinance（计算） | SMA/EMA/RSI/MACD 等 |
| ETF信息 | yfinance | FMP | 持仓、费率、追踪指数 |

## 执行流程

### Step 1: 确定查询类型

根据用户需求，对照上表确定查询类型和主/备数据源。

### Step 2: 调用数据采集脚本

统一入口脚本，自动处理数据源切换：

```bash
python3 scripts/us_market_query.py --type <查询类型> --symbol <ticker> [选项]
```

**查询类型（--type）：**
- `quote` — 实时行情
- `history` — 历史K线（可选 --period 1mo/3mo/6mo/1y/5y/max, --interval 1d/1wk/1mo）
- `financials` — 财务报表（可选 --statement income/balance/cashflow）
- `profile` — 公司概况
- `analyst` — 分析师评级与目标价
- `news` — 公司新闻
- `index` — 指数行情（symbol 用 ^GSPC / ^IXIC / ^DJI）
- `dividends` — 分红记录
- `insider` — 内部交易
- `sector` — 板块热力图
- `technical` — 技术指标（可选 --indicator sma/ema/rsi/macd, --period 20）
- `etf` — ETF信息

**示例：**
```bash
# 苹果实时行情
python3 scripts/us_market_query.py --type quote --symbol AAPL

# 特斯拉最近1年日K
python3 scripts/us_market_query.py --type history --symbol TSLA --period 1y

# 英伟达财报（利润表）
python3 scripts/us_market_query.py --type financials --symbol NVDA --statement income

# 标普500指数
python3 scripts/us_market_query.py --type index --symbol ^GSPC

# 苹果分析师评级
python3 scripts/us_market_query.py --type analyst --symbol AAPL
```

### Step 3: 结果解读

脚本输出 JSON 到 `/tmp/us_market_{type}_{symbol}.json`。

读取后按以下原则向用户呈现：
- 金额用美元，大数字用 B（十亿）/ M（百万）
- 涨跌用颜色描述（涨/跌/平）
- 财报数据做同比分析
- 如有备用数据源数据，标注来源

### API Keys 配置

详见 `references/api-keys.md`。无 key 时 yfinance 和 finmap MCP 仍可用。

### 数据源故障处理

脚本内置自动 fallback 逻辑：
1. 先尝试主数据源
2. 主数据源超时/报错/限速 → 自动切换备用数据源
3. 所有源失败 → 返回错误信息和建议

详见 `references/fallback-logic.md`。
