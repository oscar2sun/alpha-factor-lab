#!/usr/bin/env python3
"""
AI投资竞赛 — 任务状态管理器
负责：任务调度状态追踪 + 断网恢复 + 待补任务队列
"""

import json
import os
import sys
import argparse
from datetime import datetime, date
from pathlib import Path

DEFAULT_STATE_FILE = "paper-trading-state.json"

# 所有定时任务定义
TASKS = {
    "quant_factor_scan": {
        "name": "因子猎人-研报扫描",
        "player": "quant",
        "frequency": "daily",
        "time": "09:00",
        "catchup": True,       # 断网恢复后补跑
        "max_catchup_days": 1  # 最多补几天前的
    },
    "quant_rebalance": {
        "name": "因子猎人-调仓",
        "player": "quant",
        "frequency": "weekly_mon",
        "time": "09:15",
        "catchup": True,
        "max_catchup_days": 3
    },
    "trader_morning": {
        "name": "技术猎手-早盘分析",
        "player": "trader",
        "frequency": "daily",
        "time": "09:20",
        "catchup": False  # 盘中数据已过时，不补
    },
    "trader_afternoon": {
        "name": "技术猎手-午盘分析",
        "player": "trader",
        "frequency": "daily",
        "time": "13:30",
        "catchup": True,
        "max_catchup_days": 1
    },
    "value_news_scan": {
        "name": "巴菲特门徒-新闻扫描",
        "player": "value",
        "frequency": "daily",
        "time": "08:30",
        "catchup": True,
        "max_catchup_days": 1
    },
    "value_deep_analysis": {
        "name": "巴菲特门徒-深度分析",
        "player": "value",
        "frequency": "weekly_fri",
        "time": "15:30",
        "catchup": True,
        "max_catchup_days": 5
    },
    "nav_update": {
        "name": "净值更新",
        "player": "all",
        "frequency": "daily",
        "time": "15:30",
        "catchup": True,
        "max_catchup_days": 1
    },
    "frontend_deploy": {
        "name": "前端部署",
        "player": "all",
        "frequency": "daily",
        "time": "16:00",
        "catchup": True,
        "max_catchup_days": 0  # 只补当天的
    }
}


def init_state() -> dict:
    """初始化任务状态"""
    return {
        "last_run": {task_id: None for task_id in TASKS},
        "pending": [],
        "status": "ok",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }


def load_state(filepath: str) -> dict:
    """加载任务状态"""
    if not os.path.exists(filepath):
        state = init_state()
        save_state(state, filepath)
        return state
    with open(filepath, 'r') as f:
        return json.load(f)


def save_state(state: dict, filepath: str):
    """保存任务状态"""
    state["updated_at"] = datetime.now().isoformat()
    with open(filepath, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def mark_done(state: dict, task_id: str, filepath: str):
    """标记任务完成"""
    today = date.today().isoformat()
    state["last_run"][task_id] = today
    # 从pending中移除
    state["pending"] = [p for p in state["pending"] if p["task_id"] != task_id or p["date"] != today]
    save_state(state, filepath)
    print(f"[OK] 任务完成: {TASKS[task_id]['name']} @ {today}")


def mark_failed(state: dict, task_id: str, reason: str, filepath: str):
    """标记任务失败（加入pending队列）"""
    today = date.today().isoformat()
    task = TASKS[task_id]
    
    if task.get("catchup", False):
        # 检查是否已在pending中
        existing = [p for p in state["pending"] if p["task_id"] == task_id and p["date"] == today]
        if not existing:
            state["pending"].append({
                "task_id": task_id,
                "name": task["name"],
                "date": today,
                "reason": reason,
                "added_at": datetime.now().isoformat()
            })
            state["status"] = "catching_up"
    
    save_state(state, filepath)
    print(f"[PENDING] 任务失败待补: {task['name']} @ {today} ({reason})")


def get_pending(state: dict) -> list:
    """获取待补任务列表（过滤掉过期的）"""
    today = date.today()
    valid_pending = []
    
    for p in state["pending"]:
        task = TASKS.get(p["task_id"])
        if not task:
            continue
        
        task_date = date.fromisoformat(p["date"])
        days_ago = (today - task_date).days
        max_days = task.get("max_catchup_days", 1)
        
        if days_ago <= max_days:
            valid_pending.append(p)
        else:
            print(f"[SKIP] 过期任务跳过: {p['name']} @ {p['date']} (已过{days_ago}天)")
    
    return valid_pending


def check_and_report(state: dict, filepath: str) -> dict:
    """检查任务状态，返回需要补跑的任务"""
    pending = get_pending(state)
    
    # 清理过期的pending
    state["pending"] = pending
    if not pending:
        state["status"] = "ok"
    save_state(state, filepath)
    
    return {
        "status": state["status"],
        "pending_count": len(pending),
        "pending": pending
    }


def is_trading_day(d: date = None) -> bool:
    """简单判断是否交易日（排除周末，节假日需要后续完善）"""
    if d is None:
        d = date.today()
    return d.weekday() < 5  # 0=Mon, 4=Fri


def should_run(task_id: str, state: dict) -> bool:
    """判断今天这个任务是否需要运行"""
    task = TASKS[task_id]
    today = date.today()
    
    # 非交易日不跑
    if not is_trading_day(today):
        return False
    
    # 今天已经跑过了
    last_run = state["last_run"].get(task_id)
    if last_run == today.isoformat():
        return False
    
    # 频率检查
    freq = task["frequency"]
    if freq == "daily":
        return True
    elif freq == "weekly_mon":
        return today.weekday() == 0  # Monday
    elif freq == "weekly_fri":
        return today.weekday() == 4  # Friday
    
    return True


# ─── CLI ───
def main():
    parser = argparse.ArgumentParser(description="AI投资竞赛 · 任务状态管理")
    subparsers = parser.add_subparsers(dest="command")
    
    # init
    sp_init = subparsers.add_parser("init", help="初始化状态文件")
    sp_init.add_argument("--output", default=DEFAULT_STATE_FILE)
    
    # done
    sp_done = subparsers.add_parser("done", help="标记任务完成")
    sp_done.add_argument("--task", required=True, choices=list(TASKS.keys()))
    sp_done.add_argument("--state", default=DEFAULT_STATE_FILE)
    
    # fail
    sp_fail = subparsers.add_parser("fail", help="标记任务失败")
    sp_fail.add_argument("--task", required=True, choices=list(TASKS.keys()))
    sp_fail.add_argument("--reason", default="unknown")
    sp_fail.add_argument("--state", default=DEFAULT_STATE_FILE)
    
    # check
    sp_check = subparsers.add_parser("check", help="检查待补任务")
    sp_check.add_argument("--state", default=DEFAULT_STATE_FILE)
    
    # should-run
    sp_sr = subparsers.add_parser("should-run", help="检查任务是否应该运行")
    sp_sr.add_argument("--task", required=True, choices=list(TASKS.keys()))
    sp_sr.add_argument("--state", default=DEFAULT_STATE_FILE)
    
    # status
    sp_status = subparsers.add_parser("status", help="输出完整状态")
    sp_status.add_argument("--state", default=DEFAULT_STATE_FILE)
    
    args = parser.parse_args()
    
    if args.command == "init":
        state = init_state()
        save_state(state, args.output)
        print(f"[OK] 状态文件已初始化: {args.output}")
    
    elif args.command == "done":
        state = load_state(args.state)
        mark_done(state, args.task, args.state)
    
    elif args.command == "fail":
        state = load_state(args.state)
        mark_failed(state, args.task, args.reason, args.state)
    
    elif args.command == "check":
        state = load_state(args.state)
        result = check_and_report(state, args.state)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.command == "should-run":
        state = load_state(args.state)
        should = should_run(args.task, state)
        print("yes" if should else "no")
        sys.exit(0 if should else 1)
    
    elif args.command == "status":
        state = load_state(args.state)
        print(f"状态: {state['status']}")
        print(f"待补任务: {len(state['pending'])}")
        print(f"\n最近运行:")
        for tid, last in state["last_run"].items():
            task = TASKS[tid]
            print(f"  {task['name']}: {last or '未运行'}")
        if state["pending"]:
            print(f"\n待补队列:")
            for p in state["pending"]:
                print(f"  - {p['name']} @ {p['date']} ({p['reason']})")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
