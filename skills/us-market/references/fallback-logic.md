# 数据源 Fallback 逻辑

## 通用故障检测

以下情况触发 fallback：
1. **HTTP 错误**：4xx/5xx 状态码
2. **超时**：请求超过 10 秒
3. **限速**：429 Too Many Requests 或 yfinance YFRateLimitError
4. **空数据**：返回为空或关键字段缺失
5. **异常**：任何未捕获的异常

## 各数据源特殊处理

### yfinance
- YFRateLimitError → 等待 2 秒重试一次 → 仍失败则 fallback
- 数据为空 DataFrame → fallback
- 注意：yfinance 没有 API key，限速后只能等

### Alpha Vantage
- "Note" 字段出现（=超出免费额度）→ fallback
- "Error Message" → symbol 不存在
- 每日 25 次限额用完 → 当天不再使用该源

### Finnhub
- 403 → API key 无效
- 429 → 超出 60次/分钟 → 等 1 秒重试

### FMP
- 403 → API key 无效或额度用完
- 返回空数组 → 该公司可能不支持

## Fallback 链

```
quote:      yfinance → Finnhub → Alpha Vantage
history:    yfinance → Alpha Vantage
financials: yfinance → FMP
profile:    yfinance → FMP
analyst:    yfinance → Finnhub
news:       Finnhub → web_search
index:      yfinance → Alpha Vantage
dividends:  yfinance → FMP
insider:    Finnhub（无备用）
sector:     finmap MCP（无备用）
technical:  Alpha Vantage → yfinance 本地计算
etf:        yfinance → FMP
```
