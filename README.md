# SRWM: Structured Reflection & World Modeling for ALFWorld Agents

<p align="center">
  <strong>Research Codebase for Reproducible Agentic Decision-Making Experiments</strong><br/>
  <em>面向论文复现、消融研究与可观测性分析的一体化实验仓库</em>
</p>

---

## 项目定位（Repository Scope）

本仓库对应论文的**核心实验实现**，聚焦于 ALFWorld 环境中的 Agent 架构设计与系统性消融分析。  
当前对外展示与维护的主线包括：

- **结构化思维决策（ToT-style action selection）**
- **失败反思机制（Structured Reflection）**
- **显式世界模型（World Model read/update）**
- **统一消融框架（Ablation Protocol）**
- **本地/在线可观测性（Local Trace + LangSmith）**

> 说明：按你的论文展示需求，仓库中的 `rl/` 目录不作为本 README 的介绍范围。

---

## 方法总览（Method Overview）

我们将 Agent 执行过程拆解为可组合模块，并通过统一控制开关进行贡献验证：

1. **Action Selection**：基于候选动作与上下文进行选择。
2. **World Model**：维护环境事实状态，支持读取与更新。
3. **Reflection**：失败后生成结构化反思，指导下一次尝试。
4. **Observability**：记录细粒度运行事件，支持离线与在线诊断。
5. **Evaluation**：从 trajectory 自动聚合成功率与失败原因等指标。

这一设计使得“算法贡献 → 运行行为 → 评价指标”形成闭环，便于论文中的定量与定性分析。

---

## 代码结构（excluding `rl/`）

```text
.
├─ src/paradigm_experiments/
│  ├─ agents/                 # action_selection / reflection / world_model / parsing
│  ├─ runtime/                # ALFWorld env 接入、LLM 调用、运行配置
│  ├─ observability/          # local trace、event log、LangSmith 追踪
│  └─ evaluation/             # trajectory 状态汇总与离线报告
├─ 消融实验/
│  ├─ agent_ablation_unified.py   # 统一 Agent 实现与消融开关
│  └─ run_ablation.py             # 推荐 CLI 入口
├─ prompts/                   # prompt 配置
├─ tests/unit/                # 单元测试
├─ runs/                      # 实验输出目录（日志、轨迹、报告）
├─ base_config.yaml           # 基础运行配置
└─ README.md
```

---

## 快速开始（Quick Start）

### 1) 环境安装

建议使用 Python 3.10+：

```bash
pip install -e .[dev]
```

或最小安装：

```bash
pip install -r requirements.txt
pip install -e .
```

### 2) 运行消融实验

统一入口：`消融实验/run_ablation.py`

```bash
python 消融实验/run_ablation.py --variant full -- --num-tasks 5
python 消融实验/run_ablation.py --variant w_o_tot -- --num-tasks 5
python 消融实验/run_ablation.py --variant full --three-shot -- --num-tasks 5
```

### 3) 生成离线报告

```bash
python -m paradigm_experiments.evaluation.report --run-dir <run-dir>
```

---

## 消融协议（Ablation Protocol）

核心实现位于 `消融实验/agent_ablation_unified.py`，通过以下开关控制贡献项：

- `ABLA_TOT`：是否启用 ToT-style 决策过程
- `ABLA_REFLECT`：是否启用失败反思
- `ABLA_WORLD`：是否启用世界模型

推荐通过 `--variant` 参数管理组合，确保实验设置可复现、可对比、可追踪。

---

## 可观测性与可复现性（Observability & Reproducibility）

### 本地观测产物

每次运行会产生两类关键文件：

- `task_*.log`：面向人工阅读的执行日志
- `traces/task_<idx>_attempt_<n>.jsonl`：结构化 trajectory（`episode_start` / `step` / `episode_end`）

这使得论文中的失败案例分析、动作级误差定位、反思效果验证可以直接基于原始轨迹复盘。

### LangSmith（可选）

默认关闭，不影响离线实验。需要在线追踪时可设置：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
export LANGSMITH_PROJECT=alfworld-agentops
```

系统会以 **task attempt** 为粒度组织父子运行树，便于从实验编号直接定位完整决策链路。

---

## 评估指标（Evaluation Signals）

离线报告可聚合（包括但不限于）：

- Success Rate
- Termination Reason 分布
- Action Parse Failure 统计
- World Model 读取/更新频次

可进一步扩展为论文附录所需的 episode-level 表格、失败类别热图与错误案例切片分析。

---

## 单元测试（Unit Tests）

```bash
pytest tests/unit -q
```

测试覆盖解析、动作选择、世界模型、反思逻辑、运行配置、观测记录与报告生成等关键模块。

---

## 引用与致谢（Citation）

如果本仓库对你的研究有帮助，欢迎在论文中引用对应工作，并注明本仓库链接与版本号（commit hash），以确保可复现性。

---

## License

如未单独声明，请根据仓库后续提供的许可证文件使用本项目代码。
