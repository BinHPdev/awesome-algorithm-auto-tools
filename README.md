# Awesome Algorithm Auto Tools

> A curated collection of tools, frameworks, and resources for **AI-driven automated model training** — letting AI agents autonomously run experiments, fine-tune models, optimize hyperparameters, and evolve themselves.

Inspired by [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch), [HuggingFace Skills](https://github.com/huggingface/skills), and the broader AutoML movement.

---

## Why This List?

The paradigm is shifting: instead of manually tuning models, we now have tools that let **AI agents design experiments, modify training code, evaluate results, and iterate autonomously** — while you sleep.

This repository collects the best open-source tools and frameworks that make this possible across the full training lifecycle.

---

## Table of Contents

- [Autonomous Experiment / Research Frameworks](#autonomous-experiment--research-frameworks)
- [Agent-Driven Training Skills (HuggingFace Ecosystem)](#agent-driven-training-skills-huggingface-ecosystem)
- [LLM Fine-Tuning Frameworks](#llm-fine-tuning-frameworks)
- [RL Alignment Training Frameworks (RLHF / GRPO)](#rl-alignment-training-frameworks-rlhf--grpo)
- [Automated Hyperparameter Optimization / AutoML](#automated-hyperparameter-optimization--automl)
- [Self-Evolving / Self-Play Training](#self-evolving--self-play-training)
- [Lightweight Pretraining Frameworks](#lightweight-pretraining-frameworks)
- [Experiment Tracking & Orchestration](#experiment-tracking--orchestration)
- [Benchmarks & Evaluation](#benchmarks--evaluation)
- [Coding Agents (for Training Script Development)](#coding-agents-for-training-script-development)
- [Recommended Stacks](#recommended-stacks)

---

## Autonomous Experiment / Research Frameworks

> Core idea: **AI agents autonomously design experiments, modify training code, evaluate results, and iterate.** You sleep, AI experiments.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [AutoResearch](https://github.com/karpathy/autoresearch) | AI agent runs autonomous ML experiments in a loop | 630 lines of Python, ~100 experiments overnight, 11% efficiency gain on GPT-2 training |
| [AI Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2) | Fully automated scientific discovery with agentic tree search | Hypothesis → Experiment → Paper, no human templates needed |
| [auto-ml-agent](https://github.com/Nikhil-Doye/auto-ml-agent) | LLM-orchestrated autonomous ML pipeline | End-to-end: data preprocessing → model deployment, multi-agent architecture |
| [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) | Benchmark for evaluating AI agents on ML experimentation | 13 end-to-end ML tasks from CIFAR-10 to BabyLM |
| [AutoAgent](https://github.com/HKUDS/AutoAgent) | Zero-code LLM agent framework with self-play customization | Create agents via natural language, iterative self-improvement |
| [ShinkaEvolve](https://github.com/SakanaAI) | LLM-as-mutation-operator program evolution framework | Evolves programs for scientific discovery |

## Agent-Driven Training Skills (HuggingFace Ecosystem)

> "Vibe Training" — use **natural language to drive the full model training lifecycle** through coding agents.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [HuggingFace Skills](https://github.com/huggingface/skills) | Standardized ML skill packages for coding agents | 12 skills: model training (SFT/DPO/GRPO), vision training, experiment tracking, evaluation, dataset management |
| [HuggingFace AutoTrain](https://github.com/huggingface/autotrain-advanced) | No-code training platform | Upload data → auto model selection → training → evaluation → Hub publishing |

**HF Skills covers:**
- `hugging-face-model-trainer` — Fine-tune LLMs with TRL (SFT, DPO, GRPO), 0.5B to 70B parameters
- `hugging-face-vision-trainer` — Train object detection & image classification (RTDETRv2, YOLOS, ViT)
- `hugging-face-jobs` — Run compute jobs on HF infrastructure with cost estimation
- `hugging-face-trackio` — ML experiment tracking with real-time metrics
- `hugging-face-evaluation` — Model evaluation with lighteval
- `hugging-face-datasets` — Dataset creation and management
- Compatible with: **Claude Code, OpenAI Codex, Google Gemini CLI, Cursor**

## LLM Fine-Tuning Frameworks

> The training engines. Upper-level agents (AutoResearch, HF Skills) ultimately call these frameworks to execute training.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [Unsloth](https://github.com/unslothai/unsloth) | Ultra-efficient LLM fine-tuning & RL | **2x faster, 70% less VRAM**; custom CUDA kernels; MoE 12x faster; [MCP Server available](https://github.com/OtotaO/unsloth-mcp-server) |
| [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | Flexible, production-ready fine-tuning | YAML-driven; v0.8.x: QAT, sequence parallelism, GRPO, full RLHF pipeline |
| [LlamaFactory](https://github.com/hiyouga/LlamaFactory) | Unified fine-tuning with Web UI | LlamaBoard browser UI; 100+ models; SFT/RLHF/DPO/PPO |
| [TRL](https://github.com/huggingface/trl) | HuggingFace's RL training library | SFT, DPO, GRPO, PPO, KTO, ORPO; deep Transformers/PEFT integration |
| [torchtune](https://github.com/pytorch/torchtune) | PyTorch-native fine-tuning | No extra abstractions; multi-node support (Feb 2025) |
| [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) | NVIDIA's DTensor-native training library | Day-0 HuggingFace support; single-to-multi-node scaling |

## RL Alignment Training Frameworks (RLHF / GRPO)

> 2025-2026 trend: **GRPO (Group Relative Policy Optimization) is replacing PPO** as the default alignment method — no critic model needed, simpler and more stable.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | High-performance RLHF framework on Ray + vLLM | 70B+ full tuning; PPO/DAPO/REINFORCE++; async agent RLHF; [MARTI](https://github.com/OpenRLHF/OpenRLHF) fork for multi-agent RL |
| [rLLM](https://github.com/rllm-org/rllm) | Post-training RL framework for language agents | Custom agents + environments → RL training → deployment; rLLM-FinQA-4B beats Qwen3-235B |
| [LlamaGym](https://github.com/KhoomeiK/LlamaGym) | Online RL fine-tuning for LLM agents | Define agent → create LLM → write RL loop |

## Automated Hyperparameter Optimization / AutoML

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [AgentHPO](https://arxiv.org/abs/2402.01881) | LLM-driven hyperparameter optimization | Matches/surpasses human best trials on 12 ML tasks with explainable results |
| [Optuna](https://optuna.org/) | Industry-standard HPO framework | Bayesian search, pruning, distributed execution, visualization dashboard |
| [Microsoft NNI](https://github.com/microsoft/nni) | Full AutoML toolkit | Neural Architecture Search + HPO + model compression + feature engineering |
| [W&B Sweeps](https://wandb.ai/site/sweeps/) | Automated hyperparameter search + tracking | Bayesian/Grid/Random search; Hyperband early stopping; cross-machine parallelism |

## Self-Evolving / Self-Play Training

> Core idea: **Models generate their own training data to train themselves**, reducing dependence on human annotations.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [SPIN](https://github.com/uclaml/SPIN) | Self-Play Fine-Tuning | Model plays against its previous iterations; outperforms DPO + GPT-4 preference data without extra annotations |
| [SPPO](https://uclaml.github.io/SPPO/) | Self-Play Preference Optimization | Iterative policy updates approximating Nash equilibrium with convergence guarantees |
| [Multi-Agent Evolve](https://arxiv.org/html/2510.23595v1) | One LLM plays Proposer + Solver + Judge roles | Verified improvements on math, coding, reasoning with Qwen2.5-3B |
| [Multiagent Finetuning](https://llm-multiagent-ft.github.io/) | Multi-agent society from same base model | Multi-agent iteration keeps improving where single-model self-training plateaus |
| [CORY](https://proceedings.neurips.cc/paper_files/paper/2024/) | Cooperative multi-agent RL fine-tuning | Pioneer + Observer dual-agent paradigm (NeurIPS 2024) |

## Lightweight Pretraining Frameworks

> Pair these with autonomous experiment frameworks — **fast, small-scale training is the foundation for autonomous experimentation**.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [nanochat](https://github.com/karpathy/nanochat) | Minimal LLM training harness (AutoResearch's engine) | Single GPU; tokenization → pretrain → finetune → eval → chat; GPT-2 for ~$48 |
| [Nanotron](https://github.com/huggingface/nanotron) | Minimal 3D-parallel LLM pretraining | Data + Tensor + Pipeline parallelism; scales from experiments to production |

## Experiment Tracking & Orchestration

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [Weights & Biases](https://wandb.ai/) | Experiment tracking + sweeps + model registry | Industry standard; integrates with all major frameworks |
| [MLflow 3.0](https://mlflow.org/) | Open-source experiment tracking + model serving | Self-hosted; nested experiments; model registry |
| [HF Trackio](https://github.com/huggingface/skills) | Lightweight experiment tracking in HF ecosystem | Deep integration with HF Skills; agents can read metrics and make decisions |

## Benchmarks & Evaluation

| Benchmark | Description | Key Highlight |
|-----------|-------------|---------------|
| [MLE-bench](https://arxiv.org/abs/2410.07095) | 75 Kaggle ML engineering competition tasks | Evaluates AI agents on real ML engineering: training, data prep, experiments |
| [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) | 13 end-to-end ML experimentation tasks | Stanford SNAP; Claude v3 Opus best at 37.5% |
| [MLRC-Bench](https://openreview.net/forum?id=t8Okk2PRWU) | ML Research Competition challenges | Tests novel methodology development |
| [LiveCodeBench](https://artificialanalysis.ai/evaluations/livecodebench) | Contamination-free coding benchmark | Fresh problems from LeetCode/AtCoder/Codeforces |

## Coding Agents (for Training Script Development)

> These agents don't train models directly, but can **write and debug training code**, completing the automation loop when paired with HF Skills.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [Aider](https://github.com/Aider-AI/aider) | Terminal AI pair programming | Git integration; supports Claude/GPT/DeepSeek/local models |
| [OpenHands](https://github.com/OpenHands/OpenHands) | AI-driven software development (open-source Devin) | Autonomous code editing + execution + debugging; MIT license |
| [SWE-agent](https://github.com/SWE-agent/SWE-agent) | Autonomous GitHub issue fixer | SWE-bench open-source SOTA (NeurIPS 2024) |

---

## Recommended Stacks

### Most Complete Automation
```
HuggingFace Skills + Claude Code + Unsloth + W&B
```
Natural language → Claude Code orchestrates → HF Skills calls Unsloth for training → W&B tracks experiments.

### Lightest Autonomous Research
```
AutoResearch + nanochat (single GPU)
```
Start before bed, wake up to ~100 autonomous experiment results.

### Most Flexible Production Setup
```
Axolotl / LlamaFactory + OpenRLHF + Optuna + MLflow
```
YAML-configured training + automated HPO + full experiment tracking.

---

## Trends (2026)

1. **AutoResearch Paradigm**: Karpathy proved "AI autonomously doing ML research" works with just 630 lines of code
2. **"Vibe Training"**: HF Skills enables natural-language-driven model training lifecycle
3. **GRPO > PPO**: DeepSeek's GRPO is becoming the default alignment method (no critic model, simpler, more stable)
4. **Self-Play Breakthrough**: Multi-agent self-evolution (SPIN, MAE) overcomes single-model self-training plateaus
5. **MCP Standardization**: Model Context Protocol adopted by OpenAI/Google/Microsoft as the "USB-C for AI agents"
6. **Single-GPU Research**: Unsloth + nanochat + AutoResearch enables individual developers to do serious LLM research

---

## Contributing

Contributions are welcome! Please open an issue or submit a PR if you know of tools that fit this collection.

Criteria for inclusion:
- Must be **directly usable** for automated model training workflows
- Preference for open-source projects with active maintenance
- Focus on tools that leverage AI/LLMs to automate the training process itself

## License

This curated list is released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

---

*Compiled March 2026. Project statuses may change — check individual GitHub repos for the latest.*
