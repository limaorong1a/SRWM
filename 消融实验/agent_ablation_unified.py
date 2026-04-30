"""ALFWorld 三模块消融实验主脚本。

推荐使用同目录的 run_ablation.py 在子进程设置变量后启动本文件，或手动 export。"""

import argparse
import json
import os
import re
import time
import math
import random
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore

    # 默认 load_dotenv 不覆盖已存在的环境变量；shell 里若已 export ALFWORLD_DATA，
    # 会继续用坏路径。override=True + 显式 .env 路径，保证与脚本同目录的 .env 生效。
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
except Exception:
    # python-dotenv 未安装时仍可运行：依赖用户自行 export 相关环境变量
    pass

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent
if _PROJECT_ROOT.name == "消融实验":
    _PROJECT_ROOT = _PROJECT_ROOT.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from paradigm_experiments.agents.world_model import (
    traced_world_model_init,
    traced_world_model_read,
    traced_world_model_update,
)
from paradigm_experiments.agents.action_selection import (
    build_action_candidates,
    parse_action_text,
    score_actions_with_judge,
)
from paradigm_experiments.agents.reflection import (
    analyze_failure,
    count_false_completion_claims,
)
from paradigm_experiments.agents.parsing import (
    StepRecord,
    extract_goal,
    get_admissible,
    parse_think_decision,
    parse_think_text,
    process_ob,
)
from paradigm_experiments.evaluation.state import (
    load_progress_state,
    metrics_summary_path,
    progress_state_path,
    rebuild_state_from_logs,
    save_metrics_summary,
    save_progress_state,
    trajectory_step_count,
)
from paradigm_experiments.runtime.runner_utils import (
    build_react_fewshot_block,
    load_prompts,
    parse_task_indices_arg,
    resolve_path_under_script_or_parent,
    task_name_from_gamefile,
)
from paradigm_experiments.runtime.llm import (
    build_client,
    emit_model_raw,
    llm,
    llm2,
    traced_env_step,
)
from paradigm_experiments.runtime.alfworld_env import (
    bootstrap_alfworld_env,
    create_single_game_env,
)
from paradigm_experiments.runtime.settings import TASK_PREFIXES, ablation_flag
from paradigm_experiments.observability.langsmith import EpisodeRunTree
from paradigm_experiments.observability.event_log import TaskFileLogger, safe_filename_fragment
from paradigm_experiments.observability.local_trace import LocalEpisodeTraceWriter
from paradigm_experiments.observability.recorder import JsonlTraceRecorder, TraceRecorder
from paradigm_experiments.observability.trace_schema import EpisodeTrace, StepTrace


# full variant：三项 contribution 默认全部开启
ABLA_TOT = ablation_flag("ABLA_TOT", True)
ABLA_REFLECT = ablation_flag("ABLA_REFLECT", True)
ABLA_WORLD = ablation_flag("ABLA_WORLD", True)


def run_episode(
    env,
    init_prompt: str,
    ob: str,
    info: Dict,
    client: Any,
    model: str,
    judge_client: Any,
    judge_model: str,
    max_tokens: int,
    failure_context: str = "",
    max_steps: int = 50,
    event_log: Optional[TaskFileLogger] = None,
    episode_trace: Optional[EpisodeRunTree] = None,
    episode_local_trace: Optional[EpisodeTrace] = None,
    trace_recorder: Optional[TraceRecorder] = None,
) -> Tuple[float, bool, List[StepRecord]]:
    """返回 (score, won, trajectory)。第二个 bool 为 ALFWorld/TextWorld 的 won，不是 env 的 episode_done。"""
    prompt = ""
    trajectory: List[StepRecord] = []
    goal = extract_goal(ob)
    task_world_model: Optional[Dict[str, Any]] = None
    current_place = "unknown"
    trace_extra = episode_trace.child_extra() if episode_trace else {}
    if ABLA_WORLD:
        world_model_init = traced_world_model_init(ob, **trace_extra)
        task_world_model = world_model_init["model"]
        current_place = str(world_model_init["current_place"])
    if event_log:
        event_log.banner("Episode 开始")
        event_log.line(f"目标(goal): {goal or '(未解析)'}")
        if ABLA_WORLD:
            event_log.line("世界模型(JSON)已初始化（仅在 need_world_model=true 时注入到 think 提示）")
            event_log.line(f"初始位置: {current_place}")
        else:
            event_log.line("[ablation] 世界模型已关闭：不维护 visited / observed 记忆，think 为单段 JSON。")
        event_log.line(f"max_steps={max_steps}，是否附带失败上下文: {'是' if failure_context else '否'}")
        if failure_context:
            event_log.line("--- 失败上下文（摘要）---")
            fc_prev = failure_context
            for ln in fc_prev.splitlines():
                event_log.line(f"  {ln}")
        event_log.line("--- 初始观测（任务描述片段）---")
        ob_prev = ob if len(ob) < 2000 else ob[:2000] + "\n…(截断)"
        for ln in ob_prev.splitlines():
            event_log.line(f"  {ln}")
        event_log.blank()

    def _emit(msg: str) -> None:
        if event_log:
            event_log.line(msg)
        else:
            print(msg)

    current_observation_for_trace = ob
    local_trace = LocalEpisodeTraceWriter(episode_local_trace, trace_recorder)
    local_trace.start(
        task=goal,
        max_steps=max_steps,
        metadata={
            "abla_tot": ABLA_TOT,
            "abla_reflect": ABLA_REFLECT,
            "abla_world": ABLA_WORLD,
            "failure_context_present": bool(failure_context),
            "initial_place": current_place,
        },
    )

    for i in range(1, max_steps + 1):
        admissible = get_admissible(info)
        think_need_world_model = False
        world_model_read = False
        world_model_update: Optional[Dict[str, Any]] = None
        if event_log:
            event_log.banner(f"步 {i}/{max_steps}")
            event_log.line(f"当前可执行动作数量: {len(admissible)}")
        admissible_preview = ", ".join(admissible[:]) if admissible else "(空)"
        last_think = ""
        for rec in reversed(trajectory):
            if rec.kind == "think" and rec.text and not rec.text.startswith("[parse_error]"):
                last_think = rec.text
                break
        false_completion_claims = count_false_completion_claims(trajectory, recent_n=10)
        think_text = ""
        if not ABLA_WORLD:
            think_prompt = (
                init_prompt
                + prompt
                + "\nOutput exactly one line starting with 'think:'. "
                + "Include your plan for the next steps. This thought will guide the next 'action'. "
                + "Do NOT output any action/command text in this line."
                + "\n【约束】若你在 think 里提到具体命令，该命令必须来自当前可执行动作列表；若不在列表中，请改为高层计划，不要杜撰命令。"
                + "\n【避免重复】不要与上一步 think 语义完全相同；若上一思路失败，请明确说明你这一步改动点。"
                + f"\n上一条有效 think: {last_think or '(无)'}"
                + f"\n目标任务(goal): {goal or '(未解析)'}"
                + f"\n当前可执行动作: {admissible_preview}"
                + "\n请仅输出一行 JSON，格式为 {\"think\":\"...\"}。"
                + "不要输出代码块、不要额外解释。"
            )
            if false_completion_claims >= 2:
                think_prompt += (
                    "\n【反循环强约束】你在最近若干步中已多次判断“任务已完成”，但环境实际尚未成功(won=False)。"
                    "请立刻回读上面的目标任务(goal)，逐项核对前置条件与最终放置条件，明确指出“还缺哪一步/哪个状态”，"
                    "并给出下一条用于补齐缺口的行动计划；禁止再次笼统声称已完成。"
                )
                if event_log:
                    event_log.line(
                        f"[anti_loop] 检测到最近误判“已完成”次数={false_completion_claims}，已注入回读目标强约束。"
                    )
            if failure_context:
                think_prompt = failure_context + "\n" + think_prompt
            last_think_output = ""
            for retry in range(2):
                think_output = llm2(
                    think_prompt,
                    client=client,
                    model=model,
                    max_tokens=max_tokens,
                    **trace_extra,
                ).strip()
                last_think_output = think_output
                think_text = parse_think_text(think_output)
                if think_text:
                    break
                emit_model_raw(_emit, f"[think][try {retry + 1}/2 解析失败]", think_output)
                if retry == 0:
                    think_prompt += (
                        "\n上一次输出无法解析为合法 think。"
                        "请二选一输出（整段只能有一种，不要混写）：\n"
                        "1) 一行以 think: 开头，例如：think: 我下一步先去 shelf 1 找花瓶。\n"
                        "2) 一行 JSON：{\"think\":\"我下一步先去 shelf 1 找花瓶。\"}\n"
                        "不要输出代码块、不要输出动作命令、不要多行。"
                    )
            if not think_text:
                emit_model_raw(_emit, "[think][最终失败] 最后一次模型原始输出", last_think_output)
                think_text = "[parse_error] think 输出解析失败"
        else:
            think_prompt = (
                init_prompt
                + prompt
                + "\nOutput one-line JSON with keys think and need_world_model."
                + "Example: {\"think\":\"...\",\"need_world_model\":false}."
                + "Include your plan for the next steps. This thought will guide the next action."
                + "Do NOT output any action/command text in think."
                + "Set need_world_model=true only when you feel confused/uncertain and need memory of visited places/items."
                + "\n【约束】若你在 think 里提到具体命令，该命令必须来自当前可执行动作列表；若不在列表中，请改为高层计划，不要杜撰命令。"
                + "\n【避免重复】不要与上一步 think 语义完全相同；若上一思路失败，请明确说明你这一步改动点。"
                + f"\n上一条有效 think: {last_think or '(无)'}"
                + f"\n目标任务(goal): {goal or '(未解析)'}"
                + f"\n当前可执行动作: {admissible_preview}"
                + "\n请仅输出一行 JSON，格式为 {\"think\":\"...\",\"need_world_model\":false}。"
                + "不要输出代码块、不要额外解释。"
            )
            if false_completion_claims >= 2:
                think_prompt += (
                    "\n【反循环强约束】你在最近若干步中已多次判断“任务已完成”，但环境实际尚未成功(won=False)。"
                    "请立刻回读上面的目标任务(goal)，逐项核对前置条件与最终放置条件，明确指出“还缺哪一步/哪个状态”，"
                    "并给出下一条用于补齐缺口的行动计划；禁止再次笼统声称已完成。"
                )
                if event_log:
                    event_log.line(
                        f"[anti_loop] 检测到最近误判“已完成”次数={false_completion_claims}，已注入回读目标强约束。"
                    )
            if failure_context:
                think_prompt = failure_context + "\n" + think_prompt
            last_think_output = ""
            for retry in range(2):
                think_output = llm2(
                    think_prompt,
                    client=client,
                    model=model,
                    max_tokens=max_tokens,
                    **trace_extra,
                ).strip()
                last_think_output = think_output
                dec = parse_think_decision(think_output)
                think_text = dec.think
                think_need_world_model = dec.need_world_model
                if think_text:
                    break
                emit_model_raw(_emit, f"[think][try {retry + 1}/2 解析失败]", think_output)
                if retry == 0:
                    think_prompt += (
                        "\n上一次输出无法解析为合法 think。"
                        "请二选一输出（整段只能有一种，不要混写）：\n"
                        "1) 一行以 think: 开头，例如：think: 我下一步先去 shelf 1 找花瓶。\n"
                        "2) 一行 JSON：{\"think\":\"我下一步先去 shelf 1 找花瓶。\",\"need_world_model\":false}\n"
                        "不要输出代码块、不要输出动作命令、不要多行。"
                    )
            if not think_text:
                emit_model_raw(_emit, "[think][最终失败] 最后一次模型原始输出", last_think_output)
                think_text = "[parse_error] think 输出解析失败"
            elif think_need_world_model and task_world_model is not None:
                wm_json = traced_world_model_read(task_world_model, **trace_extra)
                world_model_read = True
                if event_log:
                    event_log.line("[world_model] 当前步触发读取（need_world_model=true），注入 JSON 记忆。")
                    for ln in wm_json.splitlines():
                        event_log.line(f"  {ln}")
                replan_prompt = (
                    (failure_context + "\n" if failure_context else "")
                    + init_prompt
                    + prompt
                    + "\n你刚才表示需要读取记忆。下面是任务级世界模型(JSON，仅含去过地点与地点所见物体)：\n"
                    + wm_json
                    + "\n请基于该记忆重新给出当前一步思考。"
                    + "\n仅输出一行 JSON：{\"think\":\"...\",\"need_world_model\":false}。"
                    + "不要输出动作命令，不要解释。"
                )
                replan_raw = llm2(
                    replan_prompt,
                    client=client,
                    model=model,
                    max_tokens=max_tokens,
                    **trace_extra,
                ).strip()
                replan_dec = parse_think_decision(replan_raw)
                if replan_dec.think:
                    think_text = replan_dec.think
                else:
                    emit_model_raw(_emit, "[think][world_model 重规划解析失败]", replan_raw)
        trajectory.append(StepRecord(kind="think", text=think_text))
        _emit(f"[think] {think_text}")
        prompt += f" think: {think_text}\nOK.\n>"
        decision_prompt = (
            init_prompt
            + prompt
            + "\n【本段指令：动作阶段】请从本步的可执行动作中选择下一条要执行的命令。"
        )
        scores: List[float] = []
        candidates: List[Dict[str, Any]] = []
        action_candidates_json = "[]"
        if admissible:
            context = init_prompt + prompt
            if ABLA_TOT:
                scores = score_actions_with_judge(
                    admissible,
                    goal,
                    context,
                    client=judge_client,
                    model=judge_model,
                    max_tokens=max_tokens,
                    **trace_extra,
                )
            else:
                scores = [50.0] * len(admissible)
                if event_log:
                    event_log.line(
                        "[ablation] TOT(动作评判/打分) 已关闭：候选均分 50.0，主模型仍按 act_index 选动作。"
                    )
            candidates = build_action_candidates(admissible, scores)
            action_candidates_json = json.dumps(candidates, ensure_ascii=False)
            decision_prompt += (
                "\n可执行动作结构化数组（完整列表）:\n"
                + action_candidates_json
            )
        # 把“严格输出格式”放在末尾，避免被候选动作块稀释
        decision_prompt += (
            "\n\n【输出格式要求】请仅输出一行 JSON。优先格式：{\"act_index\":N}，"
            "其中 N 必须是上方结构化数组中的 index。"
            "兼容格式：{\"act\":\"...\"}，其值必须与数组中某个 cmd 逐字一致。"
            "不要输出代码块，不要输出解释，不要输出数组外命令。"
        )
        if event_log and admissible:
            event_log.line("[judge] 候选动作及分数（至多前 15 条）:")
            for j, a in enumerate(admissible[:]):
                sc = scores[j] if j < len(scores) else 0.0
                event_log.line(f"  {j+1}. {a}  |  {sc:.1f}")
        if failure_context:
            decision_prompt = failure_context + "\n" + decision_prompt
        text = ""
        output = ""
        act_raw_attempts: List[Tuple[int, str]] = []
        for retry in range(3):
            prompt_try = decision_prompt
            if retry == 1:
                prompt_try += (
                    "\n\n【纠错-1】你上一次输出不合规（不是合法 JSON 或 act 不在候选动作中）。"
                    "请严格按以下规则输出：\n"
                    "- 只输出一行 JSON 对象，不要任何其它字符（不要前后空行、不要代码块、不要解释）。\n"
                    "- 仅允许键 act_index 或 act。\n"
                    "- 若用 act_index，必须是上方结构化数组中的 index（推荐）。\n"
                    "- 若用 act，必须逐字匹配数组中某个 cmd（禁止改写）。\n"
                    f"\n候选动作结构化数组：\n{action_candidates_json}\n"
                )
            elif retry == 2:
                prompt_try += (
                    "\n\n【纠错-2（强制索引协议）】你仍未输出合法动作。现在禁止输出 act 字符串，改用索引选择。"
                    "请只输出一行 JSON：{\"act_index\":N}，其中 N 必须取自候选动作结构化数组中的 index。"
                    "禁止输出任何其它键。"
                    f"\n候选动作结构化数组：\n{action_candidates_json}\n"
                )
            output = llm(
                prompt_try,
                client=client,
                model=model,
                max_tokens=max_tokens,
                **trace_extra,
            ).strip()
            act_raw_attempts.append((retry, output))
            text = parse_action_text(output, admissible)
            if text:
                break
            emit_model_raw(_emit, f"[act][try {retry + 1}/3 解析失败]", output)
        if not text:
            local_trace.record_step(
                StepTrace(
                    step_index=i,
                    phase="act",
                    thought=think_text,
                    action_raw=output,
                    action_validated="",
                    admissible_actions=admissible,
                    observation_before=current_observation_for_trace,
                    parser_status="action_parse_failed",
                    candidates=candidates,
                    scores=scores,
                    metadata={
                        "act_attempts": len(act_raw_attempts),
                        "false_completion_claims": false_completion_claims,
                        "world_model_read": world_model_read,
                        "need_world_model": think_need_world_model,
                    },
                )
            )
            _emit("[act] 多次解析失败，汇总各次模型原始输出如下：")
            for ridx, out in act_raw_attempts:
                emit_model_raw(_emit, f"  └── [act][try {ridx + 1}/3]", out)
            if event_log:
                event_log.banner("Episode 提前结束：act 输出无法解析")
                event_log.line("return score=0.0, won=False（原因：多次 act 解析失败/不合规）")
            local_trace.finish(
                False,
                False,
                "action_parse_failed",
                false_completion_claims=count_false_completion_claims(trajectory, recent_n=20),
            )
            return 0.0, False, trajectory
        _emit(f"[act] kind=act  text={text}")
        trajectory.append(StepRecord(kind="act", text=text))
        observation, _env_reward, env_done, info = traced_env_step(env, text, **trace_extra)
        observation = process_ob(observation[0])
        won = bool(info.get("won", [False])[0])
        episode_over = bool(env_done[0])
        _emit(f"[ob]  {observation}")
        _emit(
            f"[env] episode_over={episode_over}  won={won}  "
            f"(说明: done/episode_over 仅表示回合结束，不等价于任务成功)"
        )
        if ABLA_WORLD and task_world_model is not None:
            world_model_update = traced_world_model_update(
                model=task_world_model,
                action=text,
                observation=observation,
                current_place=current_place,
                **trace_extra,
            )
            current_place = str(world_model_update["current_place"])
        trajectory.append(StepRecord(kind="ob", text=observation))
        local_trace.record_step(
            StepTrace(
                step_index=i,
                phase="env_step",
                thought=think_text,
                action_raw=output,
                action_validated=text,
                admissible_actions=admissible,
                observation_before=current_observation_for_trace,
                observation_after=observation,
                reward=float(_env_reward[0]) if isinstance(_env_reward, (list, tuple)) and _env_reward else None,
                done=episode_over,
                won=won,
                parser_status="ok",
                candidates=candidates,
                scores=scores,
                metadata={
                    "act_attempts": len(act_raw_attempts),
                    "false_completion_claims": false_completion_claims,
                    "world_model_read": world_model_read,
                    "need_world_model": think_need_world_model,
                    "world_model_update": world_model_update,
                },
            )
        )
        current_observation_for_trace = observation
        prompt += f" {text}\n{observation}\n>"
        if episode_over:
            score = 1.0 if won else 0.0
            if event_log:
                if won:
                    event_log.banner("Episode 结束：任务成功（won=True）")
                else:
                    event_log.banner(
                        "Episode 结束：回合已结束但未达成目标（won=False，多为超时或失败终止）"
                    )
                event_log.line(f"return score={score}  won={won}")
            local_trace.finish(
                bool(won),
                won,
                "success" if won else "env_done_without_win",
                false_completion_claims=count_false_completion_claims(trajectory, recent_n=20),
            )
            return score, won, trajectory
    if event_log:
        event_log.banner("Episode 结束：未在步数内完成（失败）")
        event_log.line("return score=0.0, won=False")
    local_trace.finish(
        False,
        False,
        "max_steps",
        false_completion_claims=count_false_completion_claims(trajectory, recent_n=20),
    )
    return 0.0, False, trajectory


def _stratified_subset_indices_1based(
    config: Dict,
    split: str,
    prefixes: Dict[str, str],
    num_tasks: int,
    fraction: float,
    seed: int,
) -> List[int]:
    """按六类任务前缀分层随机抽样，返回 1-based task indices（升序去重）。"""
    if fraction <= 0:
        return []
    if fraction >= 1:
        return list(range(1, num_tasks + 1))

    # 预扫：需要一个独立 sampler_env，避免消耗主循环的 reset 序列
    _, scan_env = bootstrap_alfworld_env(config, split)
    try:
        by_prefix: Dict[str, List[int]] = {k: [] for k in prefixes.keys()}
        for idx0 in range(num_tasks):
            _ob_raw, info = scan_env.reset()
            game_file = info["extra.gamefile"][0]
            name = task_name_from_gamefile(game_file)
            matched = False
            for k in prefixes.keys():
                if name.startswith(k):
                    by_prefix[k].append(idx0 + 1)  # 1-based
                    matched = True
                    break
            if not matched:
                # 未匹配到六类前缀的关卡：不参与分层抽样
                pass
    finally:
        try:
            scan_env.close()
        except Exception:
            pass

    rng = random.Random(int(seed))
    chosen: List[int] = []
    for k, idxs in by_prefix.items():
        n = len(idxs)
        if n <= 0:
            continue
        m = int(math.ceil(n * float(fraction)))
        m = max(1, min(n, m))
        chosen.extend(rng.sample(idxs, m))
    chosen = sorted(set(chosen))
    return chosen

def main():
    parser = argparse.ArgumentParser(description="Idea3 ALFWorld runner")
    parser.add_argument(
        "--failure-retries",
        type=int,
        default=None,
        help=(
            "同一关卡下，首次未成功后最多再跑完整 episode 的次数（每次对该关卡 task_env.reset() 重开）。"
            "总尝试次数 = 1 + 该值。默认 1；0 表示不重试。"
        ),
    )
    parser.add_argument(
        "--event-dir",
        type=str,
        default=None,
        help="任务过程日志根目录；每任务仅一个 .log（含失败重试全流程）。默认 idea3/event/run_<时间>_<pid>/。",
    )
    parser.add_argument(
        "--no-event-log",
        action="store_true",
        help="关闭写入 event 日志（仍保留控制台流程输出）。",
    )
    parser.add_argument(
        "--no-event-echo",
        action="store_true",
        help="写日志文件时不再逐行同步 print（仅写文件）。",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        metavar="N",
        help=(
            "最多串行跑多少个评测关卡（每次 sampler_env.reset() 取下一关）。"
            "默认跑满 eval 集（134）；测架构时可设小值如 3。"
            "也可用环境变量 IDEA3_NUM_TASKS。"
        ),
    )
    parser.add_argument(
        "--resume-run-dir",
        type=str,
        default=None,
        help=(
            "从指定 run 目录断点续跑（读取该目录下 progress_state.json）。"
            "会从 next_task_idx 继续，并复用同一目录写日志与进度。"
        ),
    )
    parser.add_argument(
        "--task-indices",
        type=str,
        default=None,
        metavar="SPEC",
        help=(
            "只跑指定关卡（1-based，与日志文件名 task_001_* 一致）。"
            "逗号分隔，支持区间，例如: 1,3,5-8,12 。"
            "与 --resume-run-dir 互斥。也可用环境变量 IDEA3_TASK_INDICES。"
        ),
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=None,
        metavar="F",
        help=(
            "按六类任务前缀分层抽样子集比例（例如 0.2 表示每类抽 1/5）。"
            "抽样在程序内进行，并转换为 --task-indices。"
            "与 --task-indices/--resume-run-dir 互斥。默认关闭。"
        ),
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        metavar="S",
        help="分层抽样随机种子（保证不同消融变体跑同一批关卡）。默认 20260422。",
    )
    args = parser.parse_args()
    print(
        "[ablation] 模块开关 ABLA_TOT / ABLA_REFLECT / ABLA_WORLD = "
        f"{ABLA_TOT} / {ABLA_REFLECT} / {ABLA_WORLD}（0 或 false 为关闭；与全量系统一致为全 1）"
    )
    if args.failure_retries is not None:
        failure_retries = max(0, args.failure_retries)
    else:
        failure_retries = max(0, int(os.getenv("IDEA3_FAILURE_RETRIES", "1")))
    num_attempts = 1 + failure_retries

    # eval_out_of_distribution 默认 134 关；限制上界避免 reset 越界行为未定义
    _default_eval_task_count = 134
    if args.num_tasks is not None:
        num_tasks = max(1, args.num_tasks)
    else:
        num_tasks = max(1, int(os.getenv("IDEA3_NUM_TASKS", str(_default_eval_task_count))))
    if num_tasks > _default_eval_task_count:
        print(
            f"[warn] --num-tasks/IDEA3_NUM_TASKS={num_tasks} 超过评测集规模 {_default_eval_task_count}，"
            f"已截断为 {_default_eval_task_count}。"
        )
        num_tasks = _default_eval_task_count

    print(f"[run] 本回合关卡数: {num_tasks}（评测集共 {_default_eval_task_count} 关）")

    task_indices_1based: Optional[List[int]] = None
    ti_raw = (args.task_indices or os.getenv("IDEA3_TASK_INDICES", "") or "").strip()
    if ti_raw:
        if args.resume_run_dir:
            raise SystemExit("[error] --task-indices / IDEA3_TASK_INDICES 与 --resume-run-dir 不能同时使用。")
        task_indices_1based = parse_task_indices_arg(ti_raw, num_tasks)
        if not task_indices_1based:
            raise SystemExit("[error] --task-indices / IDEA3_TASK_INDICES 解析结果为空。")
        print(f"[run] 子集模式：共 {len(task_indices_1based)} 关 → {task_indices_1based}")

    # metrics 分母：子集为子集长度，否则为 num_tasks
    metrics_num_tasks_target = len(task_indices_1based) if task_indices_1based else num_tasks

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(_script_dir) == "消融实验":
        # base_config.yaml 与 prompts/ 位于上级 paradigm_experiments/
        base_dir = os.path.dirname(_script_dir)
        event_root = os.path.join(_script_dir, "event")
    else:
        base_dir = _script_dir
        event_root = os.path.join(base_dir, "event")
    prefixes = TASK_PREFIXES
    event_session_dir: Optional[str] = None
    start_idx = 0
    subset_start_0b = 0
    st_resume: Dict[str, Any] = {}
    loaded_rs: Optional[List[float]] = None
    loaded_cnts: Optional[List[int]] = None
    loaded_success_counts: Optional[List[int]] = None
    loaded_success_step_sums: Optional[List[float]] = None
    resume_dir: Optional[str] = None
    if args.resume_run_dir:
        rel = args.resume_run_dir
        if os.path.isabs(rel):
            resume_dir = rel
        else:
            resume_dir = resolve_path_under_script_or_parent(rel, _script_dir, base_dir)
        has_any_task_logs = bool(glob(os.path.join(resume_dir, "task_*.log")))
        p_prog = progress_state_path(resume_dir)
        if os.path.exists(p_prog):
            try:
                with open(p_prog, "r", encoding="utf-8") as f:
                    st_resume = json.load(f) or {}
            except Exception:
                st_resume = {}
        if not isinstance(st_resume, dict):
            st_resume = {}
        ti0 = st_resume.get("task_indices_1based")
        if isinstance(ti0, list) and len(ti0) > 0:
            task_indices_1based = [int(x) for x in ti0]
            metrics_num_tasks_target = len(task_indices_1based)
            if st_resume.get("next_subset_i") is not None:
                subset_start_0b = max(0, int(st_resume["next_subset_i"]))
        is_sub_ti0 = isinstance(ti0, list) and len(ti0) > 0
        if has_any_task_logs and (not is_sub_ti0):
            rebuilt = rebuild_state_from_logs(resume_dir, num_tasks, prefixes)
            start_idx = max(0, min(int(rebuilt.get("next_task_idx", 0)), num_tasks))
            loaded_rs = [float(x) for x in rebuilt.get("rs", [0.0] * 6)]
            loaded_cnts = [int(x) for x in rebuilt.get("cnts", [0] * 6)]
            loaded_success_counts = [int(x) for x in rebuilt.get("success_counts", [0] * 6)]
            loaded_success_step_sums = [float(x) for x in rebuilt.get("success_step_sums", [0.0] * 6)]
            print("[resume] 使用 run 目录中的 task 日志重建断点状态。")
        else:
            if not os.path.exists(p_prog) and (not (isinstance(ti0, list) and len(ti0) > 0)) and (not has_any_task_logs):
                raise FileNotFoundError(f"resume 目录中未找到 progress_state.json: {p_prog}")
            if os.path.exists(p_prog):
                st_resume = load_progress_state(resume_dir)
            state = st_resume
            ti0 = state.get("task_indices_1based", ti0)
            if (
                state.get("next_subset_i") is None
                and isinstance(ti0, list)
                and len(ti0) > 0
                and os.path.exists(metrics_summary_path(resume_dir))
            ):
                try:
                    with open(metrics_summary_path(resume_dir), "r", encoding="utf-8") as f:
                        mjk = json.load(f)
                    if isinstance(mjk, dict) and mjk.get("num_tasks_target") and int(
                        mjk.get("num_tasks_target", 0) or 0
                    ) < 134 and "next_task_idx" in mjk:
                        subset_start_0b = max(0, int(mjk.get("next_task_idx", 0) or 0))
                except Exception:
                    pass
            state_num_tasks = int(state.get("num_tasks", num_tasks))
            if state_num_tasks != num_tasks:
                print(
                    f"[resume] 检测到历史 num_tasks={state_num_tasks}，当前参数 num_tasks={num_tasks}。"
                    f"为保证一致性，采用历史值 {state_num_tasks}。"
                )
                num_tasks = state_num_tasks
            if isinstance(ti0, list) and len(ti0) > 0 and task_indices_1based:
                if state.get("next_subset_i") is not None:
                    subset_start_0b = max(0, int(state.get("next_subset_i", 0) or 0))
                if loaded_rs is None and state.get("rs") is not None:
                    loaded_rs = [float(x) for x in state.get("rs", [0.0] * 6)]
                if loaded_cnts is None and state.get("cnts") is not None:
                    loaded_cnts = [int(x) for x in state.get("cnts", [0] * 6)]
                if loaded_success_counts is None and state.get("success_counts") is not None:
                    loaded_success_counts = [int(x) for x in state.get("success_counts", [0] * 6)]
                if loaded_success_step_sums is None and state.get("success_step_sums") is not None:
                    loaded_success_step_sums = [float(x) for x in state.get("success_step_sums", [0.0] * 6)]
                n_sub = len(task_indices_1based) if task_indices_1based else 0
                if subset_start_0b < n_sub:
                    print(
                        f"[resume] 子集续跑：下标 next_subset_i={subset_start_0b} / 子集共 {n_sub} 关；"
                        f"全局 num_tasks={num_tasks}"
                    )
            else:
                start_idx = max(0, min(int(state.get("next_task_idx", 0)), num_tasks))
                if loaded_rs is None:
                    loaded_rs = [float(x) for x in state.get("rs", [0.0] * 6)]
                if loaded_cnts is None:
                    loaded_cnts = [int(x) for x in state.get("cnts", [0] * 6)]
                if loaded_success_counts is None:
                    loaded_success_counts = [int(x) for x in state.get("success_counts", [0] * 6)]
                if loaded_success_step_sums is None:
                    loaded_success_step_sums = [float(x) for x in state.get("success_step_sums", [0.0] * 6)]
            print(
                f"[resume] 使用 progress_state.json（子集/全量以 task_indices_1based 为准）。"
            )
        event_session_dir = resume_dir
        print(f"[resume] 从目录恢复: {event_session_dir}")
        if not task_indices_1based:
            print(f"[resume] 断点位置: 全局 next_task_idx={start_idx} / {num_tasks}")
    elif not args.no_event_log:
        rel = args.event_dir or event_root
        event_session_dir = os.path.join(
            rel if os.path.isabs(rel) else os.path.join(_script_dir, rel),
            f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}",
        )
        os.makedirs(event_session_dir, exist_ok=True)
        print(f"[event] 本次运行日志目录: {event_session_dir}")
    event_echo = not args.no_event_echo
    with open(os.path.join(base_dir, "base_config.yaml"), "r", encoding="utf-8") as reader:
        config = yaml.safe_load(reader)
    split = os.getenv("ALFWORLD_SPLIT", "eval_out_of_distribution")

    # 分层抽样：必须在创建主 sampler_env 前完成（避免 reset 序列被消耗）
    s_seed: Optional[int] = None
    s_frac: Optional[float] = None
    if st_resume:
        if st_resume.get("subset_seed") is not None:
            s_seed = int(st_resume["subset_seed"])
        if st_resume.get("subset_fraction") is not None:
            s_frac = float(st_resume["subset_fraction"])
    if task_indices_1based is None:
        subset_fraction = args.subset_fraction
        if subset_fraction is None:
            subset_fraction = float(os.getenv("IDEA3_SUBSET_FRACTION", "0") or "0")
        subset_seed = args.subset_seed
        if subset_seed is None:
            subset_seed = int(os.getenv("IDEA3_SUBSET_SEED", "20260422"))
        if s_frac is None and float(subset_fraction) > 0:
            s_frac = float(subset_fraction)
        if s_seed is None and s_frac and s_frac > 0:
            s_seed = int(subset_seed)
        if s_frac and s_frac > 0:
            task_indices_1based = _stratified_subset_indices_1based(
                config=config,
                split=split,
                prefixes=prefixes,
                num_tasks=num_tasks,
                fraction=float(s_frac),
                seed=int(s_seed) if s_seed is not None else 0,
            )
            if not task_indices_1based:
                raise SystemExit("[error] 分层抽样得到的 task_indices 为空，请检查 fraction/num_tasks。")
            metrics_num_tasks_target = len(task_indices_1based)
            print(
                f"[subset] 分层抽样：fraction={float(s_frac):.3f} seed={int(s_seed) if s_seed is not None else 0} "
                f"→ 共 {len(task_indices_1based)} 关"
            )
            print(f"[subset] task_indices_1based={task_indices_1based}")
    n_sub, a0 = 0, 0
    if task_indices_1based:
        metrics_num_tasks_target = len(task_indices_1based)
        n_sub = int(len(task_indices_1based))
        a0 = min(max(0, int(subset_start_0b)), n_sub)

    _, sampler_env = bootstrap_alfworld_env(config, split)
    prompt_file = os.getenv(
        "ALFWORLD_PROMPT_FILE",
        os.path.join(base_dir, "prompts", "alfworld_3prompts.json"),
    )
    prompts = load_prompts(prompt_file)
    client = build_client("OPENAI_API_KEY")
    judge_client = build_client("QNAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "qwen3-235b-a22b")
    judge_model = os.getenv("JUDGE_MODEL", model)
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
    fewshot_shots = max(0, int(os.getenv("IDEA3_FEWSHOT_SHOTS", "0")))
    cnts = loaded_cnts if loaded_cnts is not None and len(loaded_cnts) == 6 else [0] * 6
    rs = loaded_rs if loaded_rs is not None and len(loaded_rs) == 6 else [0.0] * 6
    success_counts = (
        loaded_success_counts if loaded_success_counts is not None and len(loaded_success_counts) == 6 else [0] * 6
    )
    success_step_sums = (
        loaded_success_step_sums
        if loaded_success_step_sums is not None and len(loaded_success_step_sums) == 6
        else [0.0] * 6
    )
    if event_session_dir:
        n_sub_i = len(task_indices_1based) if task_indices_1based else 0
        save_progress_state(
            run_dir=event_session_dir,
            num_tasks=num_tasks,
            next_task_idx=start_idx,
            rs=rs,
            cnts=cnts,
            success_counts=success_counts,
            success_step_sums=success_step_sums,
            failure_retries=failure_retries,
            split=split,
            task_indices_1based=task_indices_1based,
            next_subset_i=(a0 if n_sub_i else None),
            subset_fraction=s_frac,
            subset_seed=s_seed,
        )
        save_metrics_summary(
            run_dir=event_session_dir,
            num_tasks=metrics_num_tasks_target,
            next_task_idx=(a0 if n_sub else start_idx),
            prefixes=prefixes,
            cnts=cnts,
            success_counts=success_counts,
            success_step_sums=success_step_sums,
        )
    if task_indices_1based is None:
        if start_idx > 0:
            print(f"[resume] 正在快进 sampler_env 到第 {start_idx + 1} 个任务...")
            for _ in range(start_idx):
                sampler_env.reset()
        main_iter = list(range(start_idx, num_tasks))
    else:
        main_iter = [int(task_indices_1based[i]) - 1 for i in range(a0, n_sub)] if n_sub else []

    subset_next_1based = 1
    for loop_i, idx in enumerate(main_iter):
        if task_indices_1based is not None and task_indices_1based:
            pos0 = a0 + int(loop_i) if n_sub else int(loop_i)
            T = int(task_indices_1based[pos0])
            while subset_next_1based < T:
                sampler_env.reset()
                subset_next_1based += 1
            ob_raw, info = sampler_env.reset()
            subset_next_1based = T + 1
        else:
            ob_raw, info = sampler_env.reset()
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])
        r = 0.0
        task_env = None
        event_log: Optional[TaskFileLogger] = None
        try:
            if event_session_dir:
                stem = f"task_{idx + 1:03d}_{safe_filename_fragment(name)}"
                log_path = os.path.join(event_session_dir, f"{stem}.log")
                event_log = TaskFileLogger(log_path, echo=event_echo)
                if task_indices_1based is not None and n_sub:
                    event_log.banner(
                        f"任务 子集 {a0 + loop_i + 1}/{n_sub} · 全局第 {idx + 1}/{num_tasks} 关"
                    )
                else:
                    event_log.banner(f"任务 #{idx + 1}/{num_tasks}")
                event_log.line(f"关卡路径名: {name}")
                event_log.line(f"game_file: {info['extra.gamefile'][0]}")
                event_log.line(f"failure_retries={failure_retries} → 每关最多尝试 {num_attempts} 次")
                event_log.line(f"model={model}  judge_model={judge_model}")
                event_log.blank()
                event_log.line("--- sampler_env.reset() 原始观测 ---")
                raw0 = ob_raw[0] if isinstance(ob_raw, (list, tuple)) and ob_raw else ob_raw
                raw_preview = str(raw0) if len(str(raw0)) < 4000 else str(raw0)[:4000] + "\n…(截断)"
                for ln in raw_preview.splitlines():
                    event_log.line(f"  {ln}")
                event_log.blank()
            else:
                print(ob_raw)

            matched = False
            for i, (k, v) in enumerate(prefixes.items()):
                if name.startswith(k):
                    matched = True
                    game_file = info["extra.gamefile"][0]
                    if event_log:
                        event_log.line(f"任务类型前缀匹配: {k} → few-shot 变体 react_{v}_*")
                        event_log.blank()
                    task_env = create_single_game_env(game_file, config)
                    fewshot_block = build_react_fewshot_block(
                        prompts=prompts,
                        variant=v,
                        shot_count=fewshot_shots,
                    )
                    prompt = (
                        "Interact with a household to solve a task. Here are examples.\n"
                        + fewshot_block
                        + "\nHere is the task.\n"
                    )
                    failure_context = ""
                    won_episode = False
                    won_steps: Optional[int] = None
                    trajectory: List[StepRecord] = []
                    attempts_meta: List[Dict[str, object]] = []
                    for attempt in range(num_attempts):
                        if event_log:
                            event_log.banner(
                                f"整关尝试 attempt {attempt + 1}/{num_attempts}（同一 game_file 上 task_env.reset）"
                            )
                        ob_t, info_t = task_env.reset()
                        ob_text = "\n".join(ob_t[0].split("\n\n")[1:])
                        init_prompt_k = prompt + ob_text + "\n>"
                        task_goal = extract_goal(ob_text)
                        trace_recorder: Optional[JsonlTraceRecorder] = None
                        episode_local_trace: Optional[EpisodeTrace] = None
                        if event_session_dir:
                            trace_path = os.path.join(
                                event_session_dir,
                                "traces",
                                f"task_{idx + 1:03d}_attempt_{attempt + 1:02d}.jsonl",
                            )
                            trace_recorder = JsonlTraceRecorder(trace_path)
                            episode_local_trace = EpisodeTrace(
                                task=task_goal,
                                task_type=k,
                                split=split,
                                max_steps=50,
                                metadata={
                                    "agent": "idea3_ablation_unified",
                                    "task_name": name,
                                    "game_file": game_file,
                                    "attempt": attempt + 1,
                                    "max_attempts": num_attempts,
                                    "model": model,
                                    "judge_model": judge_model,
                                    "fewshot_shots": fewshot_shots,
                                },
                            )
                            if event_log:
                                event_log.line(f"[trace] 本次尝试 JSONL 轨迹: {trace_path}")
                        try:
                            with EpisodeRunTree(
                                name=f"alfworld_episode_task_{idx + 1:03d}_attempt_{attempt + 1}",
                                inputs={
                                    "task": task_goal,
                                    "task_name": name,
                                    "game_file": game_file,
                                },
                                metadata={
                                    "agent": "idea3_ablation_unified",
                                    "attempt": attempt + 1,
                                    "max_attempts": num_attempts,
                                    "model": model,
                                    "judge_model": judge_model,
                                    "abla_tot": ABLA_TOT,
                                    "abla_reflect": ABLA_REFLECT,
                                    "abla_world": ABLA_WORLD,
                                },
                            ) as episode_trace:
                                r, won_episode, trajectory = run_episode(
                                    task_env,
                                    init_prompt_k,
                                    ob_text,
                                    info_t,
                                    client=client,
                                    model=model,
                                    judge_client=judge_client,
                                    judge_model=judge_model,
                                    max_tokens=max_tokens,
                                    failure_context=failure_context,
                                    event_log=event_log,
                                    episode_trace=episode_trace,
                                    episode_local_trace=episode_local_trace,
                                    trace_recorder=trace_recorder,
                                )
                                if (not won_episode) and attempt < num_attempts - 1 and ABLA_REFLECT:
                                    if event_log:
                                        event_log.banner("本关本次尝试失败，准备失败分析并重试")
                                    failure_context = analyze_failure(
                                        trajectory=trajectory,
                                        goal=task_goal,
                                        client=client,
                                        model=model,
                                        max_tokens=max_tokens,
                                        event_log=event_log,
                                        **episode_trace.child_extra(),
                                    )
                                episode_trace.end(
                                    outputs={
                                        "score": r,
                                        "won": won_episode,
                                        "trajectory_records": len(trajectory),
                                        "episode_steps": trajectory_step_count(trajectory),
                                        "failure_reflection": bool(
                                            (not won_episode) and attempt < num_attempts - 1 and ABLA_REFLECT
                                        ),
                                    }
                                )
                        finally:
                            if trace_recorder is not None:
                                trace_recorder.close()
                        step_count = trajectory_step_count(trajectory)
                        attempts_meta.append(
                            {
                                "attempt": attempt + 1,
                                "reward": r,
                                "won": won_episode,
                                "steps_in_trajectory": len(trajectory),
                                "episode_steps": step_count,
                            }
                        )
                        if won_episode:
                            won_steps = step_count
                            if event_log:
                                event_log.banner("本关最终结果：成功（won=True）")
                                event_log.line(
                                    f"成功于第 {attempt + 1} 次尝试，score={r}，成功步数={won_steps}"
                                )
                            break
                        if attempt < num_attempts - 1:
                            if not ABLA_REFLECT:
                                failure_context = ""
                                if event_log:
                                    event_log.banner("本关本次尝试失败，准备重试（消融：结构化反思已关闭，不注入分析）")
                    if not won_episode and event_log:
                        event_log.banner("本关最终结果：在允许尝试次数内仍未成功")
                        event_log.line(f"最后 score={r}")
                    rs[i] += r
                    cnts[i] += 1
                    if won_episode:
                        success_counts[i] += 1
                        success_step_sums[i] += float(won_steps if won_steps is not None else 0.0)
                    if event_log:
                        event_log.banner("=== 任务摘要（单文件：上文已含各 attempt 步级流水 + 重试）===")
                        event_log.line(f"idx={idx + 1}  name={name}")
                        event_log.line(f"game_file={game_file}")
                        event_log.line(f"task_prefix={k}  fewshot_variant={v}")
                        event_log.line(
                            f"final_score={r}  final_won={won_episode}  "
                            f"configured_attempts={num_attempts}  actual_attempts={len(attempts_meta)}"
                        )
                        for am in attempts_meta:
                            event_log.line(
                                f"  - attempt {am['attempt']}: score={am['reward']}  "
                                f"won={am['won']}  trajectory_records={am['steps_in_trajectory']}  "
                                f"episode_steps={am['episode_steps']}"
                            )
                        if won_episode:
                            event_log.line(f"success_episode_steps={won_steps}")
                    break

            if not matched:
                msg = f"未匹配任何任务类型前缀，跳过本关 (name={name})"
                print(msg)
                if event_log:
                    event_log.line(msg)
                    event_log.banner("=== 任务摘要 ===")
                    event_log.line(f"idx={idx + 1}  skipped=True  reason=no_prefix_match  name={name}")
        finally:
            if task_env is not None:
                task_env.close()
            if event_log is not None:
                event_log.banner(f"任务 #{idx + 1} 日志结束")
                event_log.close()
        tail = f"r={r}  rs={rs}  cnts={cnts}  mean={sum(rs) / max(1, sum(cnts)):.4f}"
        if task_indices_1based is not None and n_sub:
            print(f"[进度 子集 {a0 + loop_i + 1}/{n_sub} · 全局第 {idx + 1}/{num_tasks} 关] {tail}")
        else:
            print(f"[进度 {idx + 1}/{num_tasks}] {tail}")
        if event_session_dir:
            save_progress_state(
                run_dir=event_session_dir,
                num_tasks=num_tasks,
                next_task_idx=idx + 1,
                rs=rs,
                cnts=cnts,
                success_counts=success_counts,
                success_step_sums=success_step_sums,
                failure_retries=failure_retries,
                split=split,
                task_indices_1based=task_indices_1based,
                next_subset_i=((a0 + loop_i + 1) if n_sub else None),
                subset_fraction=s_frac,
                subset_seed=s_seed,
            )
            if task_indices_1based is not None and n_sub:
                m_next = a0 + loop_i + 1
                m_num = n_sub
            else:
                m_next = idx + 1
                m_num = num_tasks
            save_metrics_summary(
                run_dir=event_session_dir,
                num_tasks=m_num,
                next_task_idx=m_next,
                prefixes=prefixes,
                cnts=cnts,
                success_counts=success_counts,
                success_step_sums=success_step_sums,
            )


if __name__ == "__main__":
    main()
