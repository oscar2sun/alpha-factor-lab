---
name: paper-trading
description: AI投资竞赛模拟交易系统。三位AI选手（因子猎人/技术猎手/巴菲特门徒）各1000万本金实时对决。当需要执行模拟交易、更新净值、查看竞赛状态、调仓决策、记录决策日志时使用。也用于竞赛相关的cron任务调度和断网恢复。
---

# Paper Trading — AI投资竞赛

三位AI选手，三种投资流派，1000万本金实时对决。

## 架构

```
paper-trading/
├── scripts/
│   ├── trading_engine.py   # 核心交易引擎（下单/撮合/净值/排行榜）
│   └── task_state.py       # 任务状态管理（调度/断网恢复/补跑）
└── references/
    ├── player_quant.md     # 因子猎人决策指南
    ├── player_trader.md    # 技术猎手决策指南
    └── player_value.md     # 巴菲特门徒决策指南
```

## 数据文件

工作目录：`alpha-factor-lab/`

| 文件 | 用途 |
|------|------|
| `paper-trading-data.json` | 竞赛核心数据（三选手portfolio/trades/nav/decisions） |
| `paper-trading-state.json` | 任务调度状态（last_run/pending队列） |

## 交易引擎 CLI

```bash
cd alpha-factor-lab

# 初始化竞赛
python3 skills/paper-trading/scripts/trading_engine.py init --start-date 2026-02-24 --cash 10000000

# 执行交易
python3 skills/paper-trading/scripts/trading_engine.py trade \
  --player quant --code 000001 --name 平安银行 \
  --price 12.5 --volume 10000 --direction buy \
  --date 2026-02-24 --reason "Amihud因子排名第1"

# 更新净值（需要价格JSON文件）
python3 skills/paper-trading/scripts/trading_engine.py nav \
  --prices-file /tmp/prices.json --benchmark-nav 1.002 --date 2026-02-24

# 查看竞赛摘要
python3 skills/paper-trading/scripts/trading_engine.py summary
python3 skills/paper-trading/scripts/trading_engine.py summary --player quant

# 记录决策
python3 skills/paper-trading/scripts/trading_engine.py decision \
  --player trader --date 2026-02-24 --type technical_analysis \
  --summary "MACD金叉+RSI超卖，买入信号" --action buy
```

## 任务状态 CLI

```bash
# 初始化
python3 skills/paper-trading/scripts/task_state.py init

# 标记完成/失败
python3 skills/paper-trading/scripts/task_state.py done --task quant_factor_scan
python3 skills/paper-trading/scripts/task_state.py fail --task nav_update --reason "断网"

# 检查待补任务
python3 skills/paper-trading/scripts/task_state.py check

# 查看完整状态
python3 skills/paper-trading/scripts/task_state.py status

# 判断是否应该运行
python3 skills/paper-trading/scripts/task_state.py should-run --task quant_rebalance
```

## 三位选手策略

### 🔢 因子猎人（quant）
- **股票池**：中证1000
- **选股**：Amihud非流动性因子排名选前20~30只
- **调仓**：每周一，等权配置
- **日常**：每天扫描研报/论文 → 构造新因子 → 回测 → 如果IC/Sharpe超过现有因子则替换
- **决策参考**：读 `references/player_quant.md`

### 📊 技术猎手（trader）
- **标的**：自动筛选1只高波动+高换手标的
- **分析**：每天2次（早盘9:20、午盘13:30）
- **指标**：MACD/KDJ/RSI/BOLL/量价关系/均线系统
- **仓位**：0%/50%/100% 三档
- **决策参考**：读 `references/player_trader.md`

### 🏛️ 巴菲特门徒（value）
- **股票池**：全A（偏好龙头蓝筹）
- **持仓**：5~10只，按conviction加权
- **日常**：每天读新闻，每周五深度分析
- **调仓**：月度或事件驱动
- **选股**：ROE>15%、PE合理、护城河深、管理层优
- **决策参考**：读 `references/player_value.md`

## Cron任务执行流程

每个cron任务的标准流程：

1. 检查网络：`timeout 5 curl -s ifconfig.me`
2. 检查是否应该运行：`task_state.py should-run --task <task_id>`
3. 如果有网且应该运行 → 执行任务
4. 完成 → `task_state.py done --task <task_id>`
5. 失败/断网 → `task_state.py fail --task <task_id> --reason "..."`

### 断网恢复

在 HEARTBEAT.md 中检查 `task_state.py check`：
- 如果有pending任务 + 当前有网 → 补跑
- 补跑完成 → 飞书通知孙瑞

## 前端数据

前端 `paper-trading.html` 读取 `paper-trading-data.json`（加cache-bust `?v=timestamp`）。

⚠️ **每次写入 decision / trade / nav 后必须立刻 git push！** 不要等frontend-deploy cron。

```bash
cd alpha-factor-lab
git add paper-trading-data.json
git commit -m "data: 竞赛数据更新 <简短描述>"
git push
```

GitHub Pages 一般1~2分钟内自动部署。不需要额外的deploy步骤。

## 费率设置
- 佣金：万三（双向），最低¥5
- 印花税：万五（仅卖出）
- 过户费：十万分之一
- 模拟滑点：千一

## 依赖的其他Skill
- **alpha-factor-lab**：因子猎人的因子挖掘和回测
- **technical-analysis**：技术猎手的K线图表和技术分析
- **buffett-analysis**：巴菲特门徒的基本面分析
- **us-market**：数据获取（如需美股标的对比）
