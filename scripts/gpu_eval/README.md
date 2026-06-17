# GPU 并行评测脚本

按数据集分子目录，避免与其它任务混淆：

| 子目录 | 数据集 | 说明 |
|--------|--------|------|
| [`gsm8k/`](gsm8k/) | GSM8K 文本四状态 | `text_gsm8k`，结果在 `results/models/text_gsm8k/` |
| [`lcb/`](lcb/) | LiveCodeBench 代码四状态 | `code_lcb`，结果在 `results/models/code_lcb/` |
| [`tool_taubench/`](tool_taubench/) | τ-bench 工具四状态 (v6) | `tool_taubench` |
| [`osworld/`](osworld/) | OSWorld GUI 四状态 | `gui_osworld_v4` |

## text_gsm8k（GSM8K）

```bash
cd /home/test/test12/songzijun/AgentReliabilityBench
bash scripts/gpu_eval/gsm8k/launch_nohup.sh
```

详见 [`gsm8k/README.md`](gsm8k/README.md)。

## code_lcb（LiveCodeBench）

```bash
bash scripts/gpu_eval/lcb/launch_nohup.sh
```

详见 [`lcb/README.md`](lcb/README.md)。

## tool_taubench（τ-bench v6）

```bash
export ARB_TOOL_BACKEND=tau
bash scripts/gpu_eval/tool_taubench/launch_nohup.sh
```

详见 [`tool_taubench/README.md`](tool_taubench/README.md)。

## gui_osworld_v4（OSWorld 模拟层）

```bash
bash scripts/gpu_eval/osworld/launch_nohup.sh
```

详见 [`osworld/README.md`](osworld/README.md)。
