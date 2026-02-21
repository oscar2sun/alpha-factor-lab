# API Keys 配置

## 免费 API Key 申请

### Alpha Vantage（推荐申请）
- 申请地址：https://www.alphavantage.co/support/#api-key
- 免费额度：25次/天
- 用途：历史股价（20年+）、技术指标

### Finnhub（推荐申请）
- 申请地址：https://finnhub.io/register
- 免费额度：60次/分钟
- 用途：实时行情、新闻、分析师评级、内部交易

### FMP（可选）
- 申请地址：https://site.financialmodelingprep.com/developer/docs
- 免费额度：250次/天
- 用途：财报、估值、SEC 文件

## 环境变量

脚本从环境变量读取 key：

```
ALPHAVANTAGE_API_KEY=your_key
FINNHUB_API_KEY=your_key
FMP_API_KEY=your_key
```

可写入 `~/.openclaw/workspace/alpha-factor-lab/.env` 文件。

## 无 Key 模式

即使没有任何 API key，以下功能仍可用：
- yfinance：股价、财报、公司信息、分红（完全免费，无需 key）
- finmap MCP：板块热力图（免费托管）
- web_search/web_fetch：新闻搜索

缺少 key 时只影响 Alpha Vantage / Finnhub / FMP 的备用能力。
