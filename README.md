# ALFWorld AgentOps Experiments

本项目聚焦当前 Idea3 / 结构化反思 / 世界模型 / 消融实验这条 Agent 架构。

旧 ReAct、CoT、ToT baseline 代码，以及早期分叉出来的结构化反思长脚本，不再作为维护目标。当前只保留一条主实现：

- `消融实验/agent_ablation_unified.py`：统一 Agent 架构实现，通过 `ABLA_TOT`、`ABLA_REFLECT`、`ABLA_WORLD` 控制三个 contribution。
- `消融实验/run_ablation.py`：推荐 CLI 入口，用 `--variant` 和 `--three-shot` 设置消融组合。

常用示例：

```powershell
python .\消融实验\run_ablation.py --variant full -- --num-tasks 5
python .\消融实验\run_ablation.py --variant w_o_tot -- --num-tasks 5
python .\消融实验\run_ablation.py --variant full --three-shot -- --num-tasks 5
```

## Runtime Artifacts

实验产物不再和源码混放。历史产物会归档到 `runs/archive/`，新的运行产物应逐步迁移到 `runs/` 下。当前兼容脚本如果仍输出到旧目录，`.gitignore` 会忽略这些目录：

- `event/`
- `eval_results/`
- `消融实验/event/`

当前运行目录内会同时保存两类观测产物：

- `task_*.log`：面向人工阅读的逐步日志。
- `traces/task_<idx>_attempt_<n>.jsonl`：面向离线分析的结构化 trajectory，包含 `episode_start`、逐步 `step` 和 `episode_end` 事件。

可基于本地 JSONL trace 生成离线报告：

```powershell
python -m paradigm_experiments.evaluation.report --run-dir <run-dir>
```

报告会聚合 success rate、termination reason、action parse failure、world model read/update 次数等指标。

## AgentOps Direction

当前已完成：

1. 旧 baseline 与重复长脚本已移除，统一入口收敛到 `agent_ablation_unified.py` / `run_ablation.py`。
2. `world_model`、`reflection`、`action_selection`、`evaluation.state`、`runtime`、`observability` 已拆成包内模块。
3. 已接入 LangSmith episode-level `RunTree` 和关键步骤 `@traceable`。
4. 已接入本地 JSONL trajectory 和离线报告。

后续若继续增强，可把 `run_episode` 主循环进一步拆成专门的 runner class，并根据真实 LangSmith 页面效果决定是否把关键步骤从 `@traceable` 改为手动 `RunTree`。

## LangSmith

LangSmith tracing 默认关闭，不会影响离线实验。需要观测链路时设置：

```powershell
$env:LANGSMITH_TRACING = "true"
$env:LANGSMITH_API_KEY = "<your-api-key>"
$env:LANGSMITH_PROJECT = "alfworld-agentops"
```

当前实现中，每个 ALFWorld task attempt 会创建一个手动 episode `RunTree`。它的作用是把一次任务尝试作为 LangSmith 中的父级 run，方便在同一个树下观察 think LLM、act LLM、judge 打分、world model read/update、`env.step` 和失败反思。这样看 trace 时不会只看到一堆散落的 LLM 调用，而是能按“第几个任务、第几次尝试”定位完整链路。

本项目采用 `src/` layout。开发时建议安装为 editable 包：

```powershell
pip install -e .[dev]
```
