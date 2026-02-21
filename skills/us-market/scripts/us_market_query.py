#!/usr/bin/env python3
"""
美股市场信息查询 - 统一入口（主/备数据源自动切换）
=================================================
用法:
    python3 us_market_query.py --type quote --symbol AAPL
    python3 us_market_query.py --type history --symbol TSLA --period 1y
    python3 us_market_query.py --type financials --symbol NVDA --statement income
    python3 us_market_query.py --type index --symbol ^GSPC
    python3 us_market_query.py --type analyst --symbol AAPL
    python3 us_market_query.py --type news --symbol AAPL
    python3 us_market_query.py --type dividends --symbol AAPL
    python3 us_market_query.py --type insider --symbol AAPL
    python3 us_market_query.py --type profile --symbol AAPL
    python3 us_market_query.py --type technical --symbol AAPL --indicator rsi --period 14
    python3 us_market_query.py --type etf --symbol SPY
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

# ─── 环境 ───
def load_env():
    """从 .env 文件加载环境变量"""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if not os.path.exists(env_path):
        env_path = os.path.expanduser('~/.openclaw/workspace/alpha-factor-lab/.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

load_env()

ALPHAVANTAGE_KEY = os.environ.get('ALPHAVANTAGE_API_KEY', '')
FINNHUB_KEY = os.environ.get('FINNHUB_API_KEY', '')
FMP_KEY = os.environ.get('FMP_API_KEY', '')

# ─── HTTP 工具 ───
import urllib.request
import urllib.error

def http_get_json(url, timeout=10):
    """简单 HTTP GET 返回 JSON"""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode())

# ─── yfinance 数据源 ───
def yf_quote(symbol):
    import yfinance as yf
    t = yf.Ticker(symbol)
    info = t.info
    return {
        'source': 'yfinance',
        'symbol': symbol,
        'name': info.get('shortName', info.get('longName', symbol)),
        'price': info.get('currentPrice') or info.get('regularMarketPrice'),
        'previous_close': info.get('previousClose'),
        'open': info.get('open') or info.get('regularMarketOpen'),
        'high': info.get('dayHigh') or info.get('regularMarketDayHigh'),
        'low': info.get('dayLow') or info.get('regularMarketDayLow'),
        'volume': info.get('volume') or info.get('regularMarketVolume'),
        'market_cap': info.get('marketCap'),
        'pe_ratio': info.get('trailingPE'),
        'forward_pe': info.get('forwardPE'),
        'dividend_yield': info.get('dividendYield'),
        'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
        'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
        'beta': info.get('beta'),
        'currency': info.get('currency', 'USD'),
    }

def yf_history(symbol, period='1y', interval='1d'):
    import yfinance as yf
    t = yf.Ticker(symbol)
    df = t.history(period=period, interval=interval)
    if df.empty:
        raise ValueError("Empty history data")
    records = []
    for idx, row in df.iterrows():
        records.append({
            'date': idx.strftime('%Y-%m-%d'),
            'open': round(row['Open'], 2),
            'high': round(row['High'], 2),
            'low': round(row['Low'], 2),
            'close': round(row['Close'], 2),
            'volume': int(row['Volume']),
        })
    return {
        'source': 'yfinance',
        'symbol': symbol,
        'period': period,
        'interval': interval,
        'count': len(records),
        'data': records
    }

def yf_financials(symbol, statement='income'):
    import yfinance as yf
    t = yf.Ticker(symbol)
    if statement == 'income':
        df = t.income_stmt
        label = 'income_statement'
    elif statement == 'balance':
        df = t.balance_sheet
        label = 'balance_sheet'
    elif statement == 'cashflow':
        df = t.cashflow
        label = 'cash_flow'
    else:
        raise ValueError(f"Unknown statement: {statement}")
    if df is None or df.empty:
        raise ValueError(f"No {statement} data")
    result = {}
    for col in df.columns:
        period_key = col.strftime('%Y-%m-%d')
        period_data = {}
        for idx, val in df[col].items():
            if val is not None and str(val) != 'nan':
                period_data[str(idx)] = float(val) if isinstance(val, (int, float)) else str(val)
        result[period_key] = period_data
    return {
        'source': 'yfinance',
        'symbol': symbol,
        'statement_type': label,
        'periods': list(result.keys()),
        'data': result
    }

def yf_profile(symbol):
    import yfinance as yf
    t = yf.Ticker(symbol)
    info = t.info
    return {
        'source': 'yfinance',
        'symbol': symbol,
        'name': info.get('longName', symbol),
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'country': info.get('country'),
        'website': info.get('website'),
        'description': info.get('longBusinessSummary', '')[:1000],
        'employees': info.get('fullTimeEmployees'),
        'market_cap': info.get('marketCap'),
        'exchange': info.get('exchange'),
        'currency': info.get('currency', 'USD'),
    }

def yf_analyst(symbol):
    import yfinance as yf
    t = yf.Ticker(symbol)
    info = t.info
    return {
        'source': 'yfinance',
        'symbol': symbol,
        'target_high': info.get('targetHighPrice'),
        'target_low': info.get('targetLowPrice'),
        'target_mean': info.get('targetMeanPrice'),
        'target_median': info.get('targetMedianPrice'),
        'recommendation': info.get('recommendationKey'),
        'recommendation_mean': info.get('recommendationMean'),
        'number_of_analysts': info.get('numberOfAnalystOpinions'),
        'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
    }

def yf_dividends(symbol):
    import yfinance as yf
    t = yf.Ticker(symbol)
    divs = t.dividends
    actions = t.actions
    records = []
    if divs is not None and not divs.empty:
        for idx, val in divs.items():
            records.append({'date': idx.strftime('%Y-%m-%d'), 'dividend': round(float(val), 4)})
    return {
        'source': 'yfinance',
        'symbol': symbol,
        'count': len(records),
        'dividends': records[-20:]  # 最近20条
    }

def yf_etf(symbol):
    import yfinance as yf
    t = yf.Ticker(symbol)
    info = t.info
    return {
        'source': 'yfinance',
        'symbol': symbol,
        'name': info.get('longName', symbol),
        'category': info.get('category'),
        'fund_family': info.get('fundFamily'),
        'expense_ratio': info.get('annualReportExpenseRatio'),
        'total_assets': info.get('totalAssets'),
        'nav': info.get('navPrice'),
        'yield': info.get('yield'),
        'ytd_return': info.get('ytdReturn'),
        'three_year_return': info.get('threeYearAverageReturn'),
        'five_year_return': info.get('fiveYearAverageReturn'),
        'beta': info.get('beta3Year'),
        'market_cap': info.get('marketCap'),
    }

# ─── Alpha Vantage 数据源 ───
def av_quote(symbol):
    if not ALPHAVANTAGE_KEY:
        raise ValueError("No ALPHAVANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHAVANTAGE_KEY}"
    data = http_get_json(url)
    if 'Note' in data or 'Information' in data:
        raise ValueError("Alpha Vantage rate limited")
    q = data.get('Global Quote', {})
    if not q:
        raise ValueError("Empty quote")
    return {
        'source': 'alpha_vantage',
        'symbol': q.get('01. symbol', symbol),
        'price': float(q.get('05. price', 0)),
        'change': float(q.get('09. change', 0)),
        'change_percent': q.get('10. change percent', ''),
        'volume': int(q.get('06. volume', 0)),
        'previous_close': float(q.get('08. previous close', 0)),
        'open': float(q.get('02. open', 0)),
        'high': float(q.get('03. high', 0)),
        'low': float(q.get('04. low', 0)),
    }

def av_history(symbol, period='1y', interval='1d'):
    if not ALPHAVANTAGE_KEY:
        raise ValueError("No ALPHAVANTAGE_API_KEY")
    # AV 免费版只支持非 ADJUSTED 接口
    if interval in ('1d', 'daily'):
        func = 'TIME_SERIES_DAILY'
        ts_key = 'Time Series (Daily)'
    elif interval in ('1wk', 'weekly'):
        func = 'TIME_SERIES_WEEKLY'
        ts_key = 'Weekly Time Series'
    elif interval in ('1mo', 'monthly'):
        func = 'TIME_SERIES_MONTHLY'
        ts_key = 'Monthly Time Series'
    else:
        func = 'TIME_SERIES_DAILY'
        ts_key = 'Time Series (Daily)'
    # compact 返回最近100条，full 太大容易超时
    outputsize = 'full' if period in ('5y', 'max') else 'compact'
    url = f"https://www.alphavantage.co/query?function={func}&symbol={symbol}&outputsize={outputsize}&apikey={ALPHAVANTAGE_KEY}"
    data = http_get_json(url)
    if 'Note' in data or 'Information' in data:
        raise ValueError("Alpha Vantage rate limited or premium endpoint")
    ts = data.get(ts_key, {})
    if not ts:
        raise ValueError("Empty time series")
    records = []
    for date_str, vals in sorted(ts.items(), reverse=True):
        records.append({
            'date': date_str,
            'open': float(vals.get('1. open', 0)),
            'high': float(vals.get('2. high', 0)),
            'low': float(vals.get('3. low', 0)),
            'close': float(vals.get('4. close', 0)),
            'volume': int(vals.get('5. volume', vals.get('6. volume', 0))),
        })
    # 按 period 截取
    limit_map = {'1mo': 22, '3mo': 66, '6mo': 132, '1y': 252, '2y': 504, '5y': 1260, 'max': 99999}
    limit = limit_map.get(period, 252)
    records = records[:limit]
    return {
        'source': 'alpha_vantage',
        'symbol': symbol,
        'period': period,
        'interval': interval,
        'count': len(records),
        'data': records
    }

def av_technical(symbol, indicator='sma', period=20):
    if not ALPHAVANTAGE_KEY:
        raise ValueError("No ALPHAVANTAGE_API_KEY")
    func_map = {'sma': 'SMA', 'ema': 'EMA', 'rsi': 'RSI', 'macd': 'MACD'}
    func = func_map.get(indicator.lower(), indicator.upper())
    if func == 'MACD':
        url = f"https://www.alphavantage.co/query?function=MACD&symbol={symbol}&interval=daily&series_type=close&apikey={ALPHAVANTAGE_KEY}"
    else:
        url = f"https://www.alphavantage.co/query?function={func}&symbol={symbol}&interval=daily&time_period={period}&series_type=close&apikey={ALPHAVANTAGE_KEY}"
    data = http_get_json(url)
    if 'Note' in data or 'Information' in data:
        raise ValueError("Alpha Vantage rate limited")
    # 找到 Technical Analysis 开头的 key
    ta_key = None
    for k in data:
        if 'Technical Analysis' in k:
            ta_key = k
            break
    if not ta_key:
        raise ValueError(f"No technical data for {indicator}")
    ta = data[ta_key]
    records = []
    for date_str, vals in sorted(ta.items(), reverse=True)[:60]:
        row = {'date': date_str}
        for k, v in vals.items():
            row[k] = float(v)
        records.append(row)
    return {
        'source': 'alpha_vantage',
        'symbol': symbol,
        'indicator': indicator,
        'period': period,
        'count': len(records),
        'data': records
    }

# ─── Finnhub 数据源 ───
def fh_quote(symbol):
    if not FINNHUB_KEY:
        raise ValueError("No FINNHUB_API_KEY")
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}"
    data = http_get_json(url)
    if not data.get('c'):
        raise ValueError("Empty Finnhub quote")
    return {
        'source': 'finnhub',
        'symbol': symbol,
        'price': data['c'],
        'change': data.get('d'),
        'change_percent': data.get('dp'),
        'high': data.get('h'),
        'low': data.get('l'),
        'open': data.get('o'),
        'previous_close': data.get('pc'),
        'timestamp': data.get('t'),
    }

def fh_news(symbol):
    if not FINNHUB_KEY:
        raise ValueError("No FINNHUB_API_KEY")
    from_date = datetime.now().strftime('%Y-%m-%d')
    # 最近30天
    import datetime as dt
    from_d = (dt.datetime.now() - dt.timedelta(days=30)).strftime('%Y-%m-%d')
    to_d = dt.datetime.now().strftime('%Y-%m-%d')
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_d}&to={to_d}&token={FINNHUB_KEY}"
    data = http_get_json(url)
    if not isinstance(data, list):
        raise ValueError("Invalid news data")
    news = []
    for item in data[:15]:
        news.append({
            'headline': item.get('headline', ''),
            'summary': item.get('summary', '')[:300],
            'source': item.get('source', ''),
            'url': item.get('url', ''),
            'datetime': datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d %H:%M') if item.get('datetime') else '',
            'category': item.get('category', ''),
        })
    return {
        'source': 'finnhub',
        'symbol': symbol,
        'count': len(news),
        'news': news
    }

def fh_analyst(symbol):
    if not FINNHUB_KEY:
        raise ValueError("No FINNHUB_API_KEY")
    url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={FINNHUB_KEY}"
    data = http_get_json(url)
    if not isinstance(data, list) or not data:
        raise ValueError("No analyst data")
    # 取最近几期
    recent = data[:6]
    # 也获取目标价
    pt_url = f"https://finnhub.io/api/v1/stock/price-target?symbol={symbol}&token={FINNHUB_KEY}"
    try:
        pt = http_get_json(pt_url)
    except:
        pt = {}
    return {
        'source': 'finnhub',
        'symbol': symbol,
        'target_high': pt.get('targetHigh'),
        'target_low': pt.get('targetLow'),
        'target_mean': pt.get('targetMean'),
        'target_median': pt.get('targetMedian'),
        'recommendations': recent
    }

def fh_insider(symbol):
    if not FINNHUB_KEY:
        raise ValueError("No FINNHUB_API_KEY")
    url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={symbol}&token={FINNHUB_KEY}"
    data = http_get_json(url)
    transactions = data.get('data', [])
    records = []
    for tx in transactions[:20]:
        records.append({
            'name': tx.get('name', ''),
            'share': tx.get('share', 0),
            'change': tx.get('change', 0),
            'filing_date': tx.get('filingDate', ''),
            'transaction_date': tx.get('transactionDate', ''),
            'transaction_type': tx.get('transactionCode', ''),
            'price': tx.get('transactionPrice'),
        })
    return {
        'source': 'finnhub',
        'symbol': symbol,
        'count': len(records),
        'insider_transactions': records
    }

# ─── FMP 数据源 ───
def fmp_profile(symbol):
    if not FMP_KEY:
        raise ValueError("No FMP_API_KEY")
    url = f"https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={FMP_KEY}"
    data = http_get_json(url)
    if not data:
        raise ValueError("Empty FMP profile")
    p = data[0] if isinstance(data, list) else data
    return {
        'source': 'fmp',
        'symbol': symbol,
        'name': p.get('companyName'),
        'sector': p.get('sector'),
        'industry': p.get('industry'),
        'country': p.get('country'),
        'website': p.get('website'),
        'description': (p.get('description') or '')[:1000],
        'employees': p.get('fullTimeEmployees'),
        'market_cap': p.get('mktCap'),
        'exchange': p.get('exchangeShortName'),
        'ceo': p.get('ceo'),
        'ipo_date': p.get('ipoDate'),
    }

def fmp_financials(symbol, statement='income'):
    if not FMP_KEY:
        raise ValueError("No FMP_API_KEY")
    stmt_map = {
        'income': 'income-statement',
        'balance': 'balance-sheet-statement',
        'cashflow': 'cash-flow-statement',
    }
    endpoint = stmt_map.get(statement, statement)
    url = f"https://financialmodelingprep.com/stable/{endpoint}?symbol={symbol}&period=annual&limit=4&apikey={FMP_KEY}"
    data = http_get_json(url)
    if not data:
        raise ValueError("Empty FMP financials")
    return {
        'source': 'fmp',
        'symbol': symbol,
        'statement_type': statement,
        'periods': [d.get('date', '') for d in data],
        'data': {d.get('date', ''): d for d in data}
    }

def fmp_dividends(symbol):
    if not FMP_KEY:
        raise ValueError("No FMP_API_KEY")
    url = f"https://financialmodelingprep.com/stable/historical-price-eod-dividend?symbol={symbol}&apikey={FMP_KEY}"
    data = http_get_json(url)
    hist = data.get('historical', [])
    records = [{'date': d.get('date', ''), 'dividend': d.get('adjDividend', d.get('dividend', 0))} for d in hist[:20]]
    return {
        'source': 'fmp',
        'symbol': symbol,
        'count': len(records),
        'dividends': records
    }

# ─── Fallback 调度 ───
FALLBACK_CHAINS = {
    'quote':      [yf_quote, fh_quote, av_quote],
    'history':    [yf_history, av_history],
    'financials': [yf_financials, fmp_financials],
    'profile':    [yf_profile, fmp_profile],
    'analyst':    [yf_analyst, fh_analyst],
    'news':       [fh_news],
    'index':      [yf_quote, av_quote],  # 指数用 quote 接口
    'dividends':  [yf_dividends, fmp_dividends],
    'insider':    [fh_insider],
    'technical':  [av_technical],
    'etf':        [yf_etf],
}

def run_with_fallback(query_type, **kwargs):
    chain = FALLBACK_CHAINS.get(query_type)
    if not chain:
        return {'error': f'Unknown query type: {query_type}'}
    
    errors = []
    for fn in chain:
        try:
            print(f"  → 尝试 {fn.__name__}...", flush=True)
            result = fn(**kwargs)
            if result:
                result['query_type'] = query_type
                result['query_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                return result
        except Exception as e:
            err_msg = f"{fn.__name__}: {type(e).__name__}: {e}"
            print(f"  ✗ {err_msg}", flush=True)
            errors.append(err_msg)
            time.sleep(0.5)  # 避免连续请求
    
    return {
        'error': f'All sources failed for {query_type}',
        'details': errors,
        'query_type': query_type,
        'query_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def main():
    parser = argparse.ArgumentParser(description='美股市场信息查询')
    parser.add_argument('--type', required=True, choices=[
        'quote', 'history', 'financials', 'profile', 'analyst',
        'news', 'index', 'dividends', 'insider', 'technical', 'etf', 'sector'
    ])
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--period', default='1y', help='历史数据周期: 1mo/3mo/6mo/1y/5y/max')
    parser.add_argument('--interval', default='1d', help='K线间隔: 1d/1wk/1mo')
    parser.add_argument('--statement', default='income', help='财报类型: income/balance/cashflow')
    parser.add_argument('--indicator', default='sma', help='技术指标: sma/ema/rsi/macd')
    parser.add_argument('--ind-period', default=20, type=int, help='技术指标周期')
    args = parser.parse_args()

    symbol = args.symbol.upper()
    qtype = args.type
    print(f"━━━ 查询 {symbol} [{qtype}] ━━━", flush=True)

    # 构建参数
    kwargs = {'symbol': symbol}
    if qtype == 'history':
        kwargs['period'] = args.period
        kwargs['interval'] = args.interval
    elif qtype == 'financials':
        kwargs['statement'] = args.statement
    elif qtype == 'index':
        pass  # 用 quote 接口
    elif qtype == 'technical':
        kwargs['indicator'] = args.indicator
        kwargs['period'] = args.ind_period
    elif qtype == 'sector':
        print("板块查询请使用 finmap MCP：finmap.list_sectors / finmap.list_tickers")
        return

    result = run_with_fallback(qtype, **kwargs)

    # 输出文件
    safe_sym = symbol.replace('^', 'IDX_').replace('/', '_')
    out_path = f"/tmp/us_market_{qtype}_{safe_sym}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n━━━ 完成 ━━━", flush=True)
    if 'error' in result:
        print(f"⚠️  {result['error']}")
        if 'details' in result:
            for d in result['details']:
                print(f"   - {d}")
    else:
        print(f"数据源: {result.get('source', 'unknown')}")
    print(f"输出: {out_path}")

    # 也输出关键数据摘要
    if qtype in ('quote', 'index') and 'price' in result:
        print(f"\n📊 {result.get('name', symbol)}: ${result['price']}")
        if result.get('change_percent'):
            print(f"   涨跌: {result.get('change_percent')}")
        if result.get('market_cap'):
            mc = result['market_cap']
            if mc > 1e12:
                print(f"   市值: ${mc/1e12:.2f}T")
            elif mc > 1e9:
                print(f"   市值: ${mc/1e9:.2f}B")
    elif qtype == 'history' and 'count' in result:
        print(f"\n📈 {result['count']} 条K线数据")
    elif qtype == 'financials' and 'periods' in result:
        print(f"\n📊 {len(result['periods'])} 期财报: {', '.join(result['periods'][:4])}")


if __name__ == '__main__':
    main()
