#!/usr/bin/env python3
"""
消融实验统一入口：在子进程中设置 ABLA_* 后执行 agent_ablation_unified.py，避免本进程已 import 导致环境变量不生效。

三开关（与论文/报告对应）：
  ABLA_TOT=1     动作评判师对候选可执行动作打分（ToT/评判模块）
  ABLA_REFLECT=1 失败后对轨迹做结构化统计 + LLM 反思，重试时注入
  ABLA_WORLD=1   世界模型：need_world_model 与 visit/items 记忆

变体名（--variant）：
  full            三者全开（默认）
  w_o_tot         去掉 TOT，候选动作用均分
  w_o_reflect     去掉结构化失败反思
  w_o_world       去掉世界模型（think 使用单段 JSON）
  w_o_tot_ref     无 TOT + 无反思
  w_o_tot_wm      无 TOT + 无世界模型
  w_o_ref_wm      无反思 + 无世界模型
  w_o_all         三者全关

示例：
  python run_ablation.py --variant w_o_tot -- --num-tasks 5
  python run_ablation.py --variant full --three-shot
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

_AGENT = Path(__file__).resolve().with_name("agent_ablation_unified.py")

VARIANTS: dict[str, dict[str, str]] = {
    "full": {},
    "w_o_tot": {"ABLA_TOT": "0"},
    "w_o_reflect": {"ABLA_REFLECT": "0"},
    "w_o_world": {"ABLA_WORLD": "0"},
    "w_o_tot_ref": {"ABLA_TOT": "0", "ABLA_REFLECT": "0"},
    "w_o_tot_wm": {"ABLA_TOT": "0", "ABLA_WORLD": "0"},
    "w_o_ref_wm": {"ABLA_REFLECT": "0", "ABLA_WORLD": "0"},
    "w_o_all": {"ABLA_TOT": "0", "ABLA_REFLECT": "0", "ABLA_WORLD": "0"},
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="在设置消融环境变量后运行 agent_ablation_unified.py。额外参数请放在 -- 之后。",
    )
    ap.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=sorted(VARIANTS.keys()),
        help="消融变体；默认 full 与全量三模块系统一致。",
    )
    ap.add_argument(
        "--three-shot",
        action="store_true",
        help="等价于设置 IDEA3_FEWSHOT_SHOTS=3。",
    )
    ap.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="传给 agent 的参数，建议以 -- 开头，如: -- --num-tasks 10",
    )
    args = ap.parse_args()

    for k, v in VARIANTS[args.variant].items():
        os.environ[k] = v
    if args.three_shot:
        os.environ["IDEA3_FEWSHOT_SHOTS"] = "3"
    if not _AGENT.is_file():
        raise SystemExit(f"未找到: {_AGENT}")
    # 去掉 REMAINDER 前可能多出来的 '--'
    forward = list(args.passthrough)
    if forward and forward[0] == "--":
        forward = forward[1:]
    sys.argv = [str(_AGENT), *forward]
    runpy.run_path(str(_AGENT), run_name="__main__")


if __name__ == "__main__":
    main()
