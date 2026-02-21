---
name: buffett-analysis
description: 巴菲特视角的上市公司基本面深度分析。当用户提到"分析一家公司"、"看看XX值不值得投资"、"XX的基本面怎么样"、"帮我研究一下XX"、个股分析、价值投资分析、公司估值、高管研究、管理层分析、公司战略、新闻采集、PR分析、业务出海、创新药等需求时使用。支持A股、港股、美股（NYSE/NASDAQ）。美股直接用 ticker（AAPL/TSLA/NVDA），A股用代码或中文名。自动获取财报、估值、行业数据、公司新闻、高管信息，按巴菲特投资框架出具完整分析报告。
---

# 巴菲特视角基本面分析

## 分析框架

巴菲特的投资决策框架——四大核心维度：

### 1. 🏰 护城河分析 (Economic Moat)
- **毛利率**：>40% 优秀，>60% 极强定价权
- **ROE**：连续>15% 是优质企业门槛，>20% 是护城河标志
- **ROIC**：>15% 说明资本配置效率高
- **应收账款周转**：越快说明议价能力越强

### 2. 📊 盈利质量 (Earnings Quality)
- **经营现金流/净利润**：>80% 说明利润是"真金白银"
- **自由现金流**：正且持续增长是好生意的标志
- **营收/利润增速**：关注稳定性而非单年爆发
- **扣非净利润 vs 净利润**：差距大说明非经常性损益多

### 3. 💰 财务健康 (Financial Health)
- **资产负债率**：因行业而异，但过高(>60%)需警惕
- **流动比率/速动比率**：>2/>1 为安全
- **有息负债**：越少越好，巴菲特偏爱低负债公司
- **货币资金/短期借款**：>1 说明现金充裕

### 4. 📐 估值合理性 (Valuation)
- **PE-TTM**：与历史分位、同行对比
- **PB**：结合ROE看（高ROE高PB合理）
- **自由现金流收益率**：FCF/市值，>5% 有吸引力
- **股息率**：巴菲特看重分红能力
- **EV/EBITDA**（美股常用）：剔除资本结构差异的估值指标，适合跨公司对比
- **股票回购**（美股重点）：美股公司常用回购代替分红，需综合考虑 Total Shareholder Yield（分红+回购）

### 5. 👔 管理层与治理 (Management & Governance)
- **高管背景**：CEO/董事长/核心高管的专业背景、行业经验
- **任职变动**：近期高管变动情况及可能原因——高管离职常是重要信号
- **管理层持股**：利益绑定程度
- **战略方向**：管理层公开言论、战略规划、业务布局

### 6. 📰 公开信息与事件驱动 (News & Catalysts)
- **公司公告/PR**：重大公告、合作协议、审批进展
- **行业新闻**：行业政策、竞争格局变化
- **媒体报道**：正面/负面报道、市场情绪
- **事件催化**：近期可能影响股价的关键事件

## 执行流程

当用户要求分析一家公司时，根据需求选择合适的分析路径：

- **完整基本面分析**：执行 Step 1-4 全流程
- **专项研究**（如高管研究、新闻采集、战略分析）：可跳过财务数据采集，直接执行 Step 1b/1c + Step 3
- **追问深化**：在已有分析基础上，补充专项研究

### Step 1a: 财务数据采集

**市场自动判别：**
- 输入为英文 ticker（如 AAPL、TSLA、NVDA、GOOGL）或明确美股公司 → 美股路径
- 输入为中文名（贵州茅台）或 A 股代码（600519、000858）→ A 股路径

#### A股/港股路径

运行脚本（使用公司名或代码）：

```bash
PATH="/home/node/.local/bin:$PATH" python3 scripts/fetch_company_data.py "贵州茅台"
```

脚本输出 JSON 到 `/tmp/buffett_analysis_{code}.json`，包含：
- 公司基本信息、行业分类
- 最新利润表、资产负债表、现金流量表
- 核心财务指标（ROE/ROIC/毛利率等，含多期对比）
- 估值指标（PE/PB/PS/PCF）
- 实时行情、市值
- 十大股东
- 同行业公司列表（用于对比）

如需同行对比数据，对2-3家可比公司额外运行：
```bash
PATH="/home/node/.local/bin:$PATH" python3 scripts/fetch_company_data.py "五粮液"
```

#### 美股路径

运行美股采集脚本：

```bash
python3 scripts/fetch_us_company_data.py AAPL
```

脚本依次调用 us-market skill 采集 profile/financials/quote/analyst/dividends，输出到 `/tmp/buffett_analysis_{TICKER}.json`。

如需同行对比：
```bash
python3 scripts/fetch_us_company_data.py MSFT
```

### Step 1b: 新闻/公开信息采集

#### A股路径

使用 `web_search` 和 `fintool-search`（MCP）采集公司相关新闻和公开信息：

1. **web_search** 搜索公司最新新闻、PR稿件、公告：
   - 搜索关键词组合：`"{公司名}" {具体话题}`（如 `"恒瑞医药" 创新药出海`）
   - 用 `freshness: "pm"` 限定近期新闻
   - 搜索 3-5 组不同关键词，覆盖不同角度

2. **web_fetch** 对重要搜索结果抓取全文

3. **fintool-search**（MCP）获取财经新闻：
   ```
   fintool-search.search_news keyword="{公司名}" count=20
   ```

#### 美股路径

1. 用 us-market skill 获取新闻：
   ```bash
   python3 skills/us-market/scripts/us_market_query.py --type news --symbol AAPL
   ```
   输出在 `/tmp/us_market_news_{TICKER}.json`

2. **web_search** 补充搜索（英文关键词）：
   - `"{Company Name}" earnings report 2026`
   - `"{Company Name}" SEC filing latest`

3. **web_fetch** 抓取关键报道全文

**信息整理要求：**
- 按时间线整理关键事件
- 区分事实（公告/官方信息）和观点（分析师/媒体评论）
- 标注信息来源和日期
- 识别信息之间的关联性

### Step 1c: 高管与管理层研究

#### A股路径

1. **web_search** 搜索高管信息：
   - `"{公司名}" 高管 任命/变动/离职`
   - `"{高管姓名}" 背景/简历`
   - `"{公司名}" 董事会 管理层`

2. **web_fetch** 抓取关键人物报道

#### 美股路径

1. 用 us-market skill 获取内部交易数据：
   ```bash
   python3 skills/us-market/scripts/us_market_query.py --type insider --symbol AAPL
   ```
   输出在 `/tmp/us_market_insider_{TICKER}.json`，含高管买卖记录

2. **web_search** 搜索高管信息（英文）：
   - `"{Company Name}" CEO management team`
   - `"{Company Name}" executive changes leadership`
   - `"{CEO Name}" background interview`

3. **web_fetch** 抓取关键报道

#### 分析要点（通用）
   - 核心高管名单（董事长、CEO、CFO、业务负责人）
   - 每位高管的教育背景、职业履历、专业领域
   - 近期任职变动：谁走了？谁来了？为什么？
   - 高管变动对公司战略方向的潜在影响
   - 管理层持股变化（增持/减持信号）

### Step 2: 读取分析模板

```
read references/analysis-template.md
```

### Step 3: 撰写分析报告

结合采集数据 + 分析模板 + 以下原则撰写报告：

**巴菲特语言风格：**
- 用生活化比喻解释财务概念
- 适当引用巴菲特/芒格名言
- 给出明确观点，不做墙头草
- 永远提示风险

**分析深度要求：**
- 不只列数字，要解读数字背后的商业逻辑
- 结合行业竞争格局分析护城河
- 横向对比（vs 同行）+ 纵向对比（vs 自身历史）
- 最终给出清晰的投资价值判断

**输出格式：**
- 飞书/群聊用 bullet points，不用 markdown 表格
- 先给结论（一句话总结），再展开分析
- 控制在合理篇幅（不要写论文）

## Step 4: 写入阿尔法工坊前端

每次完成基本面分析后，**必须**将报告数据写入前端展示：

1. 读取 `alpha-factor-lab/fundamental-reports.json`
2. 按以下 JSON 结构追加一条报告：

```json
{
  "name": "公司名称",
  "code": "600519.SH 或 AAPL（美股直接用ticker）",
  "market": "A 或 US",
  "date": "2026-02-21",
  "rating": "推荐|强烈推荐|中性|回避|关注",
  "moat": {
    "score": 8,
    "summary": "护城河分析文字...",
    "metrics": { "毛利率": "91.3%", "ROE": "36.99%", ... }
  },
  "earnings": {
    "score": 7,
    "summary": "盈利质量分析文字...",
    "metrics": { ... }
  },
  "health": {
    "score": 9,
    "summary": "财务健康分析文字...",
    "metrics": { ... }
  },
  "valuation": {
    "score": 6,
    "summary": "估值分析文字...",
    "metrics": { "PE-TTM": "20.66x", ... }
  },
  "management": {
    "score": 7,
    "summary": "管理层与治理分析文字...",
    "key_people": ["姓名 - 职位 - 简要背景", ...],
    "recent_changes": ["变动描述1", ...]
  },
  "catalysts": {
    "score": 7,
    "summary": "近期动态与催化分析文字...",
    "events": ["事件1（日期）", ...],
    "outlook": "前瞻判断文字..."
  },
  "conclusion": "综合结论文字...",
  "risks": ["风险1", "风险2", ...]
}
```

3. 如果该公司已存在（按 code 匹配），则**覆盖更新**；否则追加
4. 写入后 commit 并 push 到 GitHub，确保阿尔法工坊在线页面同步更新

**阿尔法工坊地址：** https://oscar2sun.github.io/alpha-factor-lab/fundamental.html
