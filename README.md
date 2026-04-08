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
- [Synthetic Data Generation & Curation](#synthetic-data-generation--curation)
- [Knowledge Distillation](#knowledge-distillation)
- [Model Merging & Quantization](#model-merging--quantization)
- [Lightweight Pretraining & Distributed Training](#lightweight-pretraining--distributed-training)
- [Inference Engines (for RL Training Loops)](#inference-engines-for-rl-training-loops)
- [Multimodal Training Frameworks](#multimodal-training-frameworks)
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
| [AutoML-Agent](https://github.com/DeepAuto-AI/automl-agent) | Multi-agent LLM framework for full-pipeline AutoML (ICML 2025) | Parallel specialized agents for preprocessing, architecture design, HPO; retrieval-augmented planning |
| [auto-ml-agent](https://github.com/Nikhil-Doye/auto-ml-agent) | LLM-orchestrated autonomous ML pipeline | End-to-end: data preprocessing → model deployment, multi-agent architecture |
| [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) | Benchmark for evaluating AI agents on ML experimentation | 13 end-to-end ML tasks from CIFAR-10 to BabyLM |
| [AutoAgent](https://github.com/HKUDS/AutoAgent) | Zero-code LLM agent framework with self-play customization | Create agents via natural language, iterative self-improvement |
| [ShinkaEvolve](https://github.com/SakanaAI) | LLM-as-mutation-operator program evolution framework | Evolves programs for scientific discovery |
| [AI-Supervisor](https://arxiv.org/abs/2603.24402) | Autonomous research supervision via persistent Research World Model | Multi-agent consensus + Knowledge Graph; validates claims via GPU computation; self-correcting updates |
| [ARIS](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) | Lightweight Markdown-only skills for autonomous ML research overnight | Zero dependencies; cross-model review loops; 20+ GPU experiments per overnight run; works with any LLM agent |

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
| [LMFlow](https://github.com/OptimalScale/LMFlow) | Extensible toolkit for fine-tuning large foundation models | LISA memory-efficient training (outperforms LoRA); FlashAttention; NAACL Best Demo Paper |
| [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio) | No-code GUI framework for fine-tuning LLMs | Browser-based UI; LoRA/4-bit/8-bit; DPO/IPO/KTO; W&B integration |
| [LitGPT](https://github.com/Lightning-AI/litgpt) | 20+ high-performance LLMs with pretrain/finetune/deploy recipes | CLI-driven; powered TinyLlama project; NeurIPS 2023 LLM Efficiency Challenge |
| [InstructLab](https://github.com/instructlab) | IBM/Red Hat collaborative LLM customization via synthetic data | LAB alignment method; taxonomy-driven skill contributions; targets Granite models |

## RL Alignment Training Frameworks (RLHF / GRPO)

> 2025-2026 trend: **GRPO (Group Relative Policy Optimization) is replacing PPO** as the default alignment method — no critic model needed, simpler and more stable.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | High-performance RLHF framework on Ray + vLLM | 70B+ full tuning; PPO/DAPO/REINFORCE++; async agent RLHF |
| [verl](https://github.com/volcengine/verl) | ByteDance's Volcano Engine RL for LLMs | GRPO/PPO in few lines; 3D-HybridEngine; used by ByteDance, Alibaba Qwen, UC Berkeley, LMSys |
| [DAPO](https://github.com/BytedTsinghua-SIA/DAPO) | Open-source RL system from ByteDance Seed + Tsinghua | 50 pts on AIME 2024 with Qwen2.5-32B; 4 key stability techniques; built on verl |
| [AReaL](https://github.com/inclusionAI/AReaL) | Fully asynchronous RL for LLM reasoning (Ant Group + Tsinghua) | 2.77x speedup vs synchronous; GSPO algorithm; Ascend NPU support |
| [slime](https://github.com/THUDM/slime) | LLM post-training framework for RL scaling (GLM team) | Powers GLM-4.5/4.6/4.7/5; Megatron + SGLang; RLVE (400 verifiable environments) |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) | NVIDIA's scalable post-training RL library | GRPO, SFT, DPO, DAPO; Ray-based; Megatron Core parallelism |
| [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) | Build RL environments for LLM training | Multi-step/multi-turn environments; interoperable with NeMo RL, OpenRLHF, TRL, Unsloth |
| [rLLM](https://github.com/rllm-org/rllm) | Post-training RL framework for language agents | Custom agents + environments → RL training → deployment; rLLM-FinQA-4B beats Qwen3-235B |
| [RAGEN](https://github.com/RAGEN-AI/RAGEN) | Multi-turn RL framework for training reasoning agents | StarPO framework; 10 built-in environments; identifies "Echo Trap" instability |
| [f-GRPO](https://github.com/rhaldarpurdue/f-GRPO) | f-Divergence based GRPO for general LLM alignment | KL/Reverse KL/Pearson/Hellinger/JS divergences; superior on both RLVR (math) and safety alignment; built on Unsloth |
| [Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO) | Tree search for LLM agent RL (ICLR 2026) | 4x less rollout budget via shared prefixes; step-wise process supervision from outcome reward; tree-structured ReAct |
| [SimpleRL-Reason](https://github.com/hkust-nlp/simpleRL-reason) | Simple RL recipe for reasoning (HKUST) | DeepSeek-R1-style; 7B achieves 33.3% AIME with only 8K examples; no SFT needed |
| [SWE-RL](https://github.com/facebookresearch/swe-rl) | Meta's RL for software engineering reasoning | Llama3-SWE-RL-70B achieves 41% on SWE-bench Verified (NeurIPS 2025) |
| [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL) | RL tuning for LLM agents (UIUC + MetaGPT) | PPO-based; AgentGym environments + verl training |
| [LlamaGym](https://github.com/KhoomeiK/LlamaGym) | Online RL fine-tuning for LLM agents | Define agent → create LLM → write RL loop |
| [Reasoning Gym](https://github.com/open-thought/reasoning-gym) | Procedural reasoning environments for RLVR | 100+ tasks; NeurIPS 2025 Spotlight; unlimited controllable task generation |

## Automated Hyperparameter Optimization / AutoML

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [AgentHPO](https://arxiv.org/abs/2402.01881) | LLM-driven hyperparameter optimization | Matches/surpasses human best trials on 12 ML tasks with explainable results |
| [AutoML-Agent](https://github.com/DeepAuto-AI/automl-agent) | Multi-agent LLM framework for full-pipeline AutoML (ICML 2025) | Parallel specialized agents; retrieval-augmented planning; 14 datasets tested |
| [Optuna](https://optuna.org/) | Industry-standard HPO framework | Bayesian search, pruning, distributed execution, visualization dashboard |
| [Microsoft NNI](https://github.com/microsoft/nni) | Full AutoML toolkit | Neural Architecture Search + HPO + model compression + feature engineering |
| [W&B Sweeps](https://wandb.ai/site/sweeps/) | Automated hyperparameter search + tracking | Bayesian/Grid/Random search; Hyperband early stopping; cross-machine parallelism |

## Self-Evolving / Self-Play Training

> Core idea: **Models generate their own training data to train themselves**, reducing dependence on human annotations.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [SPIN](https://github.com/uclaml/SPIN) | Self-Play Fine-Tuning | Model plays against its previous iterations; outperforms DPO + GPT-4 preference data without extra annotations |
| [SPPO](https://uclaml.github.io/SPPO/) | Self-Play Preference Optimization | Iterative policy updates approximating Nash equilibrium with convergence guarantees |
| [SPC (Self-Play Critic)](https://chen-judge.github.io/SPC/) | Adversarial self-play for evolving reasoning critics | "Sneaky generator" vs "critic" game; eliminates manual step-level annotation |
| [SPELL](https://arxiv.org/html/2509.23863) | Self-Play RL for Evolving Long-Context Language Models | Label-free self-play; base model surpasses instruction-tuned counterpart on long-context tasks |
| [Multi-Agent Evolve](https://arxiv.org/html/2510.23595v1) | One LLM plays Proposer + Solver + Judge roles | Verified improvements on math, coding, reasoning with Qwen2.5-3B |
| [Multiagent Finetuning](https://llm-multiagent-ft.github.io/) | Multi-agent society from same base model | Multi-agent iteration keeps improving where single-model self-training plateaus |
| [CORY](https://proceedings.neurips.cc/paper_files/paper/2024/) | Cooperative multi-agent RL fine-tuning | Pioneer + Observer dual-agent paradigm (NeurIPS 2024) |

## Synthetic Data Generation & Curation

> Critical for automated training pipelines: **generate high-quality training data at scale** without manual annotation.

### Data Generation

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [Distilabel](https://github.com/argilla-io/distilabel) | Framework for synthetic data and AI feedback pipelines | Modular pipeline; SFT/DPO/UltraFeedback techniques; any LLM provider |
| [Magpie](https://github.com/magpie-align/magpie) | Alignment data synthesis from scratch (ICLR 2025) | No prompt engineering needed; 4M instructions generated; matches Llama-3 Instruct |
| [DataDreamer](https://github.com/datadreamer-dev/DataDreamer) | Reproducible synthetic data generation (ACL 2024) | Multi-step prompting; generate/align/fine-tune/distill; built-in caching |
| [Cosmopedia](https://github.com/huggingface/cosmopedia) | Large-scale synthetic pretraining data pipeline | 25B tokens of synthetic textbooks/blogs; uses Mixtral-8x7B |
| [InstructLab SDG](https://github.com/instructlab/sdg) | Synthetic data via LAB methodology (IBM/Red Hat) | Skills-SDG + Knowledge-SDG; minimal seed taxonomy → large-scale data |
| [Persona Hub](https://github.com/tencent-ailab/persona-hub) | Persona-driven synthetic data at billion scale (Tencent) | 1B diverse personas; 370M elite personas released |
| [synth_gen](https://github.com/facebookresearch/synth_gen) | Execution-verified synthetic data (Meta) | Modular verifier system; parser-based verification for code |
| [Evidently](https://github.com/evidentlyai/evidently) | Open-source synthetic data generation with user profiles | Model-agnostic; customizable personas & goals; no-code UI in Evidently Cloud; outputs to pandas DataFrame |
| [NVIDIA Nemotron-4 340B](https://blogs.nvidia.com/blog/nemotron-4-synthetic-data-generation-llm-training/) | Open models for synthetic data generation pipeline | Base + Instruct + Reward models; commercial use allowed |

### Data Curation & Filtering

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) | GPU-accelerated data preprocessing & curation | 30+ filters; fuzzy dedup 1.1T tokens in 1.8h on 64 A100s; 16x faster |
| [DataTrove](https://github.com/huggingface/datatrove) | Platform-agnostic data processing pipeline | Used for FineWeb and Cosmopedia; low memory; Slurm support |
| [Dolma](https://github.com/allenai/dolma) | High-performance dataset curation toolkit (AllenAI) | Built-in parallelism for billions of docs; used for OLMo (3T tokens) |
| [Data Prep Kit](https://github.com/data-prep-kit/data-prep-kit) | Unstructured data preparation (IBM) | Python/Ray/Spark runtimes; laptop to datacenter scaling |

## Knowledge Distillation

> **Compress large models into smaller, deployable ones** while preserving capabilities.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [EasyDistill](https://github.com/modelscope/easydistill) | Comprehensive distillation toolkit (Alibaba/ModelScope, EMNLP 2025) | Black-box + white-box KD; data synthesis + SFT + logits distillation + RL |
| [DistillKit](https://github.com/arcee-ai/DistillKit) | Production-ready LLM distillation (Arcee AI) | Online and offline workflows; powers Arcee Virtuoso, SuperNova models |
| [MiniPLM](https://github.com/thu-coai/MiniPLM) | Knowledge distillation for pre-training (Tsinghua, ICLR 2025) | Improved DPKD variant |
| [DistiLLM](https://github.com/jongwooko/distillm) | Streamlined distillation with contrastive approach (ICML 2024) | DistiLLM-2 contrastive distillation |

## Model Merging & Quantization

> **Combine multiple models or compress them** for efficient deployment and training.

### Model Merging

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [MergeKit](https://github.com/arcee-ai/mergekit) | Leading toolkit for merging pretrained LLMs | SLERP, TIES, DARE, Passthrough, Evolutionary merge; works on CPU with 8GB VRAM |
| [MergeLM](https://github.com/yule-BUAA/MergeLM) | Language model merging codebase (ICML 2024) | Research-grade implementations |

### Quantization

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [GPTQModel](https://github.com/ModelCloud/GPTQModel) | Production-ready LLM quantization toolkit | GPTQ, AWQ, QQQ, GPTAQ, EoRA, GAR; multi-backend CPU/GPU |
| [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) | Easy-to-use GPTQ quantization | 8/4/3/2-bit; Marlin int4*fp16 kernel; ~150-200K monthly PyPI downloads |
| [AutoRound](https://github.com/intel/auto-round) | Advanced quantization via sign-gradient descent (Intel) | High accuracy at 2-4 bits; exports to GPTQ/AWQ/GGUF; broad HW compatibility |
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) | Unified quantization, pruning, distillation & speculative decoding | FP8/INT8/INT4; exports to TensorRT-LLM/vLLM; NeMo Megatron integration |
| [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) | Google's KV cache compression (ICLR 2026) | 6x memory reduction at 3-bit with zero accuracy loss; PolarQuant + QJL; 8x perf on H100 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | LLM inference in C/C++ with GGUF quantization | Q4_K_M sweet spot: 92% quality, 75% size reduction; runs everywhere |

## Lightweight Pretraining & Distributed Training

> Pair these with autonomous experiment frameworks — **fast, small-scale training is the foundation for autonomous experimentation**.

### Lightweight Pretraining

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [nanochat](https://github.com/karpathy/nanochat) | Minimal LLM training harness (AutoResearch's engine) | Single GPU; tokenization → pretrain → finetune → eval → chat; GPT-2 for ~$48 |
| [Nanotron](https://github.com/huggingface/nanotron) | Minimal 3D-parallel LLM pretraining | Data + Tensor + Pipeline parallelism; scales from experiments to production |

### Distributed Training

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [TorchTitan](https://github.com/pytorch/torchtitan) | PyTorch-native large-scale training platform | Up to 4D parallelism without model code changes; MXFP8 on Blackwell; elastic scaling |
| [Open-dLLM](https://github.com/pengzhangzhi/Open-dLLM) | First open-source full stack for diffusion LLMs | Raw data → training → checkpoints → evaluation → inference, all-in-one |

## Inference Engines (for RL Training Loops)

> Inference engines are critical for RL training — **80% of RLHF training time is spent on sample generation**. Fast inference = fast training.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [vLLM](https://github.com/vllm-project/vllm) | Most mature open-source LLM serving engine | PagedAttention; 4x higher throughput on Blackwell; core engine for OpenRLHF |
| [SGLang](https://github.com/sgl-project/sglang) | High-performance serving for LLMs & multimodal | ~16,200 tok/sec on H100; RadixAttention; used by slime for RL training |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA's optimized inference library | FP8/FP4/INT4; EAGLE-3 speculative decoding; max GPU performance |
| [LMDeploy](https://github.com/InternLM/lmdeploy) | LLM compression, deployment & serving | TurboMind MXFP4; 1.5x vLLM performance; DeepSeek PD disaggregation |
| [HuggingFace TGI](https://github.com/huggingface/text-generation-inference) | Multi-backend LLM serving (TensorRT-LLM, vLLM, llama.cpp) | Unified frontend; token streaming; HF Hub native; CPU/GPU/Inferentia support |
| [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) | Datacenter-scale distributed inference | 30x request throughput on DeepSeek-R1; disaggregated prefill/decode; Rust + Python |

## Multimodal Training Frameworks

> Training models that understand **text, images, video, and audio** simultaneously.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [LLaVA-OneVision-1.5](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5) | Fully open-source multimodal training | Native-resolution images; SOTA performance; lower training costs |
| [LLaVA-OneVision-1.5-RL](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5-RL) | Democratized multimodal RL training | Open code, data, and models for multimodal RLHF |
| [OpenRLHF-M](https://github.com/OpenRLHF/OpenRLHF-M) | Multimodal model RLHF training | Extension of OpenRLHF for VLMs |
| [LLaVA-KD](https://github.com/Fantasyele/LLaVA-KD) | Multimodal knowledge distillation (ICCV 2025) | Distills large MLLMs into smaller ones |
| [MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA) | Mixture-of-Experts for vision-language models (TMM 2025) | Efficient multimodal MoE architecture |

## Experiment Tracking & Orchestration

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [Weights & Biases](https://wandb.ai/) | Experiment tracking + sweeps + model registry | Industry standard; integrates with all major frameworks |
| [MLflow 3.0](https://mlflow.org/) | Open-source experiment tracking + model serving | Self-hosted; nested experiments; model registry |
| [ClearML](https://github.com/allegroai/clearml) | Open-source MLOps platform | 150K+ users at Fortune 500; auto-logging; pipeline orchestration; dataset versioning |
| [HF Trackio](https://github.com/huggingface/skills) | Lightweight experiment tracking in HF ecosystem | Deep integration with HF Skills; agents can read metrics and make decisions |

## Benchmarks & Evaluation

### ML Agent Benchmarks

| Benchmark | Description | Key Highlight |
|-----------|-------------|---------------|
| [MLE-bench](https://arxiv.org/abs/2410.07095) | 75 Kaggle ML engineering competition tasks | Evaluates AI agents on real ML engineering: training, data prep, experiments |
| [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) | 13 end-to-end ML experimentation tasks | Stanford SNAP; Claude v3 Opus best at 37.5% |
| [PaperBench](https://openai.com/index/paperbench/) | Evaluates AI's ability to replicate ICML 2024 papers | 8,316 gradable tasks across 20 papers; best agent scores 21% |
| [CORE-Bench](https://openreview.net/forum?id=BsMMc4MEGS) | Computational Reproducibility Agent Benchmark | 270 tasks from 90 papers across CS, social science, medicine |
| [MLRC-Bench](https://openreview.net/forum?id=t8Okk2PRWU) | ML Research Competition challenges | Tests novel methodology development |
| [AgentBench](https://github.com/THUDM/AgentBench) | Multi-dimensional benchmark for LLM agents | Tests across OS, database, knowledge graph, web, and game environments |
| [SWE-bench Verified](https://www.swebench.com/) | Human-verified GitHub issue resolution | Industry standard for coding agents; top scores 70%+ |
| [LiveBench](https://livebench.ai/) | Monthly-updated contamination-free LLM benchmark | 6 categories (Math/Reasoning/Coding/Language/Data/IF); objective auto-scoring; no LLM judge needed |

### Model Evaluation Frameworks

| Tool | Description | Key Highlight |
|------|-------------|---------------|
| [DeepEval](https://github.com/confident-ai/deepeval) | Pytest-like LLM evaluation framework | v3.0: 14+ metrics; multi-turn simulation; DeepTeam for red teaming |
| [Opik](https://github.com/comet-ml/opik) | Open-source LLM observability & evaluation (Comet) | Deep tracing; LLM-as-a-judge; hallucination detection; production dashboards |
| [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | Multimodal evaluation across text, image, video, audio | v0.6: eval-as-a-service; 7.5x throughput; 50+ tasks |
| [Arize Phoenix](https://github.com/Arize-ai/phoenix) | Open-source LLM observability and evaluation | Fully self-hosted; tracing, evaluation, retrieval analysis |
| [LiveCodeBench](https://artificialanalysis.ai/evaluations/livecodebench) | Contamination-free coding benchmark | Fresh problems from LeetCode/AtCoder/Codeforces |

## Coding Agents (for Training Script Development)

> These agents don't train models directly, but can **write and debug training code**, completing the automation loop when paired with HF Skills.

| Project | Description | Key Highlight |
|---------|-------------|---------------|
| [Aider](https://github.com/Aider-AI/aider) | Terminal AI pair programming | Git integration; supports Claude/GPT/DeepSeek/local models |
| [OpenHands](https://github.com/OpenHands/OpenHands) | AI-driven software development (open-source Devin) | Autonomous code editing + execution + debugging; MIT license |
| [SWE-agent](https://github.com/SWE-agent/SWE-agent) | Autonomous GitHub issue fixer | SWE-bench open-source SOTA (NeurIPS 2024) |
| [Open-SWE](https://github.com/langchain-ai/open-swe) | LangChain's async cloud-hosted coding agent | Multi-agent (Planner + Reviewer); GitHub integration; auto PR creation |
| [SERA](https://huggingface.co/collections/allenai/open-coding-agents) | Ai2's open coding agent family | 54.2% on SWE-Bench; trains in 40 GPU-days (~$2K); all open |
| [Cline](https://github.com/cline/cline) | VS Code AI coding agent with 60K+ GitHub stars | MCP tool creation; 5M+ developers; human-in-the-loop approval; native subagents |
| [OpenCode](https://github.com/opencode-ai/opencode) | Go-based terminal AI agent with 95K+ GitHub stars | Bubble Tea TUI; 75+ LLM providers; 6.5M monthly developers; SQLite persistence |
| [Plandex](https://github.com/plandex-ai/plandex) | Terminal agent for large projects with 2M token context | Tree-sitter project maps; diff review sandbox; auto-debugging; 30+ languages |
| [Roo Code](https://github.com/RooVetGit/Roo-Code) | Terminal agent with 95K+ GitHub stars | 75+ LLM providers; plan-first development; 2.5M monthly developers |

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

### Full RL Training Pipeline (2026 SOTA)
```
verl / OpenRLHF + vLLM/SGLang + Reasoning Gym + W&B
```
State-of-the-art RL training with fast inference engines and rich environments.

### Synthetic Data → Training → Eval
```
Distilabel / Magpie → Unsloth / TRL → DeepEval / LMMs-Eval
```
Generate data at scale → train efficiently → evaluate comprehensively.

---

## Trends (2026 Q2 Update)

1. **AutoResearch Paradigm**: Karpathy proved "AI autonomously doing ML research" works with just 630 lines of code — now spawning derivatives like ARIS and AI-Supervisor
2. **"Vibe Training"**: HF Skills enables natural-language-driven model training lifecycle
3. **GRPO Variants Proliferate**: f-GRPO (f-divergence family), Tree-GRPO (tree search, ICLR 2026), DAPO — GRPO is the new default, and specialized variants are emerging fast
4. **RL Framework Explosion**: verl, DAPO, AReaL, slime — every major lab now has an open-source RL training framework
5. **Self-Play Breakthrough**: Multi-agent self-evolution (SPIN, MAE, SPC) overcomes single-model self-training plateaus
6. **Synthetic Data as Infrastructure**: Distilabel, Magpie, Evidently make data generation a first-class pipeline stage; model collapse mitigation (Evol-Instruct) becoming standard
7. **MCP Standardization**: Model Context Protocol adopted by OpenAI/Google/Microsoft as the "USB-C for AI agents"
8. **Single-GPU Research**: Unsloth + nanochat + AutoResearch enables individual developers to do serious LLM research
9. **Inference-Training Convergence**: vLLM/SGLang/TGI are now core components of RL training loops, not just serving
10. **Multimodal RL**: LLaVA-OneVision-1.5-RL and OpenRLHF-M bring RL alignment to vision-language models
11. **Extreme Quantization**: Google TurboQuant achieves 6x KV cache compression at zero accuracy loss (ICLR 2026); NVIDIA Model Optimizer unifies quantization/pruning/distillation
12. **Multi-Agent Coding Wave**: Feb 2026 saw every major tool ship multi-agent capabilities (Grok Build, Windsurf, Claude Code, Codex CLI, Devin) — coding agents now routinely write training scripts

---

## Related Awesome Lists

- [Awesome LLM Synthetic Data](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data)
- [Awesome Knowledge Distillation of LLMs](https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs)
- [Awesome Model Merging](https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications)
- [Awesome LLM Quantization](https://github.com/pprp/Awesome-LLM-Quantization)
- [Awesome LLM Inference Engine](https://github.com/sihyeong/Awesome-LLM-Inference-Engine)
- [LLM Datasets](https://github.com/mlabonne/llm-datasets)
- [LLM Distillation Playbook](https://github.com/predibase/llm_distillation_playbook)

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

*Compiled March 2026, updated April 2026. Project statuses may change — check individual GitHub repos for the latest.*
