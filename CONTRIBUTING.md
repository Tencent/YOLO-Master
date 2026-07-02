<div align="center">
  <img src="https://github.com/Tencent/YOLO-Master/raw/main/.github/assets/contributing_banner.png" alt="Contributing to YOLO-Master" width="100%"/>
  <br><br>
  <img
    width="160"
    alt="YOLO-Master Mascot"
    src="https://github.com/user-attachments/assets/bbf751ea-af27-465d-a8a9-7822db343638"
  />
</div>

<h1 align="center">Contributing to YOLO-Master</h1>

<p align="center">
  <strong>Mixture-of-Experts · Dynamic Routing · Real-Time Detection</strong>
</p>

---

感谢你愿意参与 [Tencent/YOLO-Master](https://github.com/Tencent/YOLO-Master) 的建设。YOLO-Master 是**腾讯优图实验室 (Tencent Youtu Lab)** 开源的实时目标检测框架，也是 CVPR 2026 收录的首个将 **Mixture-of-Experts (MoE)** 深度融合进 YOLO 架构的工作。本项目已入选 [腾讯犀牛鸟开源计划 (Rhino Bird Open Source Program)](https://opensource.tencent.com/summer-of-code/)，我们欢迎来自社区的 Bug 报告、功能改进、文档修正、示例补充、模型复现结果和工程化优化。

> **We welcome contributions in both Chinese and English.** This guide is written primarily in Chinese with English keywords preserved for technical accuracy.

---

## Table of Contents

- [行为准则](#行为准则)
- [可以贡献什么](#可以贡献什么)
- [提交 Issue 前](#提交-issue-前)
- [报告 Bug](#报告-bug)
- [功能建议](#功能建议)
- [Pull Request 流程](#pull-request-流程)
- [本地开发](#本地开发)
- [代码规范](#代码规范)
- [测试与验证](#测试与验证)
- [文档和示例](#文档和示例)
- [模型数据和权重](#模型数据和权重)
- [PR 评审标准](#pr-评审标准)
- [许可证和版权](#许可证和版权)
- [基于 YOLO-Master 的开源项目](#基于-yolo-master-的开源项目)
- [安全问题](#安全问题)
- [常见问题](#常见问题)

---

## 行为准则

请在 Issues、Pull Requests、代码评审和社区讨论中保持 **尊重、友善、专业和建设性**。我们欢迎不同背景、经验和观点的贡献者参与，也希望所有讨论都聚焦于技术事实、可复现证据和项目目标。

请避免人身攻击、歧视性表达、骚扰、刷屏、无关推广或泄露他人隐私信息。维护者可以关闭偏离主题或不符合社区协作氛围的讨论。

---

## 可以贡献什么

- **Bug 报告**：提交可复现的问题、错误日志、环境信息和最小复现脚本。
- **功能建议**：提出与实时目标检测、MoE 动态路由、LoRA 高效微调、稀疏推理、训练流程、模型评测或部署优化相关的改进想法。
- **代码贡献**：修复缺陷、补充测试、改进性能、完善导出链路（ONNX / TensorRT / CoreML / OpenVINO / ncnn）、优化 MoE 剪枝或路由策略。
- **文档贡献**：修正文档错误，补充教程、FAQ、配置说明、中英文翻译或 MoE 原理 Wiki（`docs/` 和 `wiki/` 目录）。
- **示例贡献**：补充训练、验证、推理、导出、LoRA 微调、MoE 分析（`diagnose_model` / `prune_moe_model`）和部署示例（`examples/` 和 `lora_examples/` 目录）。
- **Docker 贡献**：改进现有 Dockerfile（GPU / CPU / ARM64 / Jetson / 导出专用），或新增平台支持。
- **Agent Skill 贡献**：完善 `yolo-master-agent` Skill Bundle 的 SKILL.md、验证套件或推理 Runner。
- **实验复现**：分享可验证的 benchmark、消融实验、硬件延迟结果或失败案例。
- **代码评审**：即使没有直接提交代码，审查他人的 PR 也是极其宝贵的贡献。

---

## 提交 Issue 前

1. 先搜索已有 [Issues](https://github.com/Tencent/YOLO-Master/issues) 和 [Pull Requests](https://github.com/Tencent/YOLO-Master/pulls)，避免重复提交。
2. 确认问题可以在最新 `main` 分支或目标发布版本上复现。
3. 如果是较大的功能或架构改动（如新增 MoE 专家数量、修改路由网络结构、新增 LoRA 适配器类型），请先开 Issue 讨论设计方向，再投入实现。
4. **不要在公开 Issue 中上传密钥、私有数据、未授权模型权重或包含敏感信息的日志。**

---

## 报告 Bug

我们高度重视 Bug 报告，因为它们直接帮助我们提升 YOLO-Master 的可靠性。一个高质量的 Bug 报告通常包含：

- 清晰的问题标题和影响范围。
- **最小可复现步骤**（Minimum Reproducible Example），包括命令、配置文件、输入数据说明或脚本片段。
- 期望行为和实际行为。
- 完整错误日志或 traceback。
- 运行环境：操作系统、Python 版本、PyTorch/TorchVision 版本、CUDA/cuDNN 或 MPS 信息、GPU/CPU 型号。
- 使用的模型、配置、权重来源和数据集信息。如果数据不能公开，请提供可替代的最小样例。

**推荐模板：**

```markdown
### 问题描述

### 复现步骤
1.
2.
3.

### 期望行为

### 实际行为 / 日志

### 环境信息
- OS:
- Python:
- PyTorch:
- CUDA / MPS:
- YOLO-Master commit:

### 其他上下文
```

---

## 功能建议

提交功能建议时，请说明：

- 这个功能解决什么问题，目标用户是谁。
- 是否已有临时方案或外部实现。
- 预期 API、配置项、命令行参数或文档入口。
- 对训练、推理、导出、兼容性和性能的影响。
- 你是否愿意提交 PR 或协助测试。

---

## Pull Request 流程

我们非常欢迎通过 [Pull Requests (PRs)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) 提交贡献。为确保评审顺利，请遵循以下步骤：

1. **[Fork 本仓库](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)**：将 [Tencent/YOLO-Master](https://github.com/Tencent/YOLO-Master) Fork 到你的 GitHub 账号。
2. **[创建分支](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop)**：基于最新 `main` 创建语义清晰的分支，例如 `fix-moe-router-shape`、`feature/moa-inference-mode`、`docs-lora-tutorial`。
3. **实施改动**：保持改动聚焦，尽量让一个 PR 只解决一个问题。确保代码遵循项目风格，不引入新的错误或警告。
4. **本地测试**：提交前在本地完成测试，确认改动不会导致 [regressions](https://en.wikipedia.org/wiki/Software_regression)。引入新功能时，请补充测试。
5. **提交 Commit**：使用简洁且描述性的 commit message。如果改动对应某个 Issue，请在 message 中引用（例如 `Fix #123: 修正动态路由梯度计算`）。
6. **创建 PR**：提交 PR 到 `Tencent/YOLO-Master:main`，并在描述中说明：
   - 改动动机与实现方式
   - 关联的 Issue 编号
   - 本地测试结果（如 `pytest` 输出、性能对比、延迟数据）
   - 是否影响现有 API 或模型兼容性

### CLA 签署

在合并你的 PR 之前，你需要签署 **Contributor License Agreement (CLA)**。这一法律协议确保你的贡献在正确的许可证下被授权，使项目能够继续以 [AGPL-3.0](https://opensource.org/license/agpl-v3) 分发。

提交 PR 后，CLA 机器人会引导你完成签署流程。请在 PR 中评论以下语句以签署：

```text
I have read the CLA Document and I sign the CLA
```

> 如果 PR 页面出现 Tencent CLA、DCO 或其他自动检查提示，请按机器人说明完成签署或确认；相关检查通过后维护者才能继续合并流程。

### GitHub Actions CI 测试

所有 PR 必须通过 [GitHub Actions](https://github.com/features/actions) 的 [Continuous Integration](https://docs.github.com/en/actions) (CI) 测试才能被合并。CI 测试包括 linting、单元测试和其他质量检查。请查看 CI 输出并修复任何报错。

---

## 本地开发

```bash
git clone https://github.com/<your-github-id>/YOLO-Master.git
cd YOLO-Master
git remote add upstream https://github.com/Tencent/YOLO-Master.git

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
```

同步上游代码：

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

**安装特定功能依赖：**

```bash
# 导出支持（ONNX / TensorRT / CoreML / OpenVINO）
pip install -e ".[export]"

# LoRA 微调支持
pip install -e ".[lora]"

# 日志追踪（WandB / TensorBoard / MLflow）
pip install -e ".[logging]"

# 完整开发环境
pip install -e ".[dev,export,lora,logging]"
```

---

## 代码规范

- **避免代码重复**：优先复用项目已有模块、配置结构和工具函数，避免引入重复实现。
- **保持改动聚焦**：以目标明确的修改为主，避免大规模重构混入无关变更。
- **简化优先**：寻找简化代码或移除冗余部分的机会。
- **向后兼容**：新增公共 API、配置项或命令行行为时，请保持向后兼容，并在 README、Wiki 或示例中说明。如果必须破坏兼容，请在 PR 中充分说明理由和迁移方案。
- **统一格式**：项目在 `pyproject.toml` 中维护了 Ruff、YAPF、isort 和 docformatter 配置。建议在提交前运行：
  ```bash
  ruff check ultralytics tests
  yapf -r -i ultralytics tests
  isort ultralytics tests
  docformatter -i ultralytics/**/*.py tests/**/*.py
  ```
- **补充测试**：引入新功能时，请补充对应测试，并说明是否覆盖 CPU / CUDA / MPS / ONNX / TensorRT 等路径。
- **敏感信息**：不要提交密钥、Token、私有路径、内部域名或未授权数据。
- **大文件**：避免把大型模型权重、生成结果、数据集、缓存文件提交到 Git。应通过 Release、对象存储或模型仓库分发。

### Google-Style Docstrings

新增函数、类或复杂逻辑时，请补充 [Google-style docstrings](https://google.github.io/styleguide/pyguide.html)。这有助于其他开发者理解和维护你的代码。项目 `pyproject.toml` 中已配置 `tool.ruff.lint.pydocstyle.convention = "google"`。

**示例：标准 Google-style**

```python
def example_function(image_path: str, conf: float = 0.25) -> bool:
    """Validate an image path before running inference.

    Args:
        image_path (str): Path to the input image.
        conf (float): Confidence threshold used by the detector.

    Returns:
        (bool): True if the image can be used for inference, False otherwise.

    Examples:
        >>> result = example_function("data/sample.jpg", 0.5)  # returns True
    """
    return bool(image_path) and conf >= 0.0
```

**示例：带类型提示的 Google-style**

```python
def route_experts(
    features: torch.Tensor,
    top_k: int = 2,
    noise_epsilon: float = 1e-2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic routing for ES-MoE block.

    Args:
        features: Input feature tensor of shape (B, C, H, W).
        top_k: Number of experts to activate per token.
        noise_epsilon: Small noise for load balancing.

    Returns:
        expert_indices: Tensor of selected expert indices.
        gate_weights: Tensor of routing weights for aggregation.

    Examples:
        >>> idx, w = route_experts(torch.randn(2, 64, 16, 16), top_k=2)
    """
    ...
```

**示例：单行 Docstring**

对于较小或简单的函数，单行 docstring 即可。必须使用三个双引号，是完整句子，首字母大写并以句号结尾。

```python
def is_moe_enabled(config: dict) -> bool:
    """Check whether MoE modules are enabled in the given config."""
    return config.get("moe", {}).get("enabled", False)
```

### MoE 模块贡献指南

由于 MoE 是 YOLO-Master 的核心创新，涉及 MoE 模块的改动需要额外注意：

- **新增专家**：确保新增专家与现有专家架构一致（遵循 `ultralytics/nn/modules/moe/experts.py` 的接口规范），并在 `default.yaml` 中补充对应配置项。
- **修改路由机制**：路由网络的改动需同时更新 `test_moe.py` 中的路由测试，确保 Top-K 选择、负载均衡和 gate 权重分布仍然正确。
- **负载均衡损失**：修改 `MoELoss` 或 `MoEPruner` 时，请说明对训练收敛和推理速度的影响，并运行 `test_mixture_aux_loss.py` 验证。
- **剪枝策略**：`prune_moe_model` 的改动需经过 `diagnose_model` 分析验证，确保专家利用率统计和剪枝后模型精度损失可接受。
- **静态图兼容性**：MoE 路由需确保 ONNX / TorchScript 导出可行（参见 `test_exports.py`）。

### LoRA 贡献指南

- LoRA 适配器应通过配置项激活，零架构侵入（zero architectural overhead）。
- 新增 LoRA 类型（如 DoRA、LoHa）时，请参照 `peft` 库接口规范，并补充 `lora_examples/` 中的示例脚本。
- 运行 `lora_e2e_smoke.py` 和 `lora_rankless_smoke.py` 进行端到端验证。

### Docker 贡献指南

项目维护多种 Docker 镜像（GPU / CPU / ARM64 / Jetson JetPack 4/5/6 / 导出专用）。贡献 Docker 时请注意：

- 基于已有 Dockerfile 进行增量修改，避免引入与官方基础镜像不兼容的依赖。
- 在 Dockerfile 顶部添加 `# YOLO-Master AGPL-3.0 License` 声明。
- 在 PR 中说明构建命令和验证步骤（如 `docker build -f Dockerfile -t yolo-master:test .`）。
- 如果新增 Dockerfile，请在 `README.md` 或 `docs/` 中补充使用说明。

---

## 测试与验证

请根据改动范围选择合适的验证方式，并在 PR 描述中粘贴关键结果。

**核心模块测试：**

```bash
# Python API 和 CLI 测试
python -m pytest tests/test_python.py tests/test_cli.py

# MoE / MoA / MoT 模块测试
python -m pytest tests/test_moe.py tests/test_moa.py tests/test_mot.py
python -m pytest tests/test_mixture_aux_loss.py tests/test_mixture_fixes.py

# CUDA 和引擎测试
python -m pytest tests/test_cuda.py tests/test_engine.py

# 导出和集成测试
python -m pytest tests/test_exports.py tests/test_integrations.py

# 解决方案和 LoRA 测试
python -m pytest tests/test_solutions.py tests/lora_e2e_smoke.py
```

**完整测试（含慢测试）：**

```bash
# 完整测试
python -m pytest tests

# 包含慢速测试（需要更多时间）
python -m pytest tests --slow

# 覆盖率报告
python -m pytest tests --cov=ultralytics --cov-report=term-missing
```

**针对改动类型的验证建议：**

| 改动类型 | 建议验证 |
|----------|----------|
| 模型结构 / MoE 模块 | `test_moe.py`, `test_moa.py`, `test_mot.py`, `diagnose_model` |
| 训练逻辑 | `test_engine.py`, `test_cuda.py` |
| 导出 / 部署 | `test_exports.py`（ONNX / TensorRT / CoreML / OpenVINO） |
| LoRA 微调 | `lora_e2e_smoke.py`, `lora_rankless_smoke.py` |
| CLI / 配置 | `test_cli.py`, `test_python.py` |
| 文档 / 示例 | 手动验证命令和链接可用性 |
| Docker | `docker build` 和 `docker run` 验证 |

项目在 `pyproject.toml` 中配置了 pytest、coverage、yapf、ruff、isort 和 docformatter。建议在提交前运行：

```bash
ruff check ultralytics tests
yapf -r -i ultralytics tests
isort ultralytics tests
```

---

## 文档和示例

文档贡献和代码贡献同样重要。提交文档或示例时，请注意：

- **中英文一致性**：中英文内容应尽量保持信息一致。
- **可运行示例**：示例命令应可以直接复制运行，必要时标注依赖、数据准备和预期输出。
- **实验可复现**：图表、benchmark 和实验结论应说明硬件、软件版本、输入尺寸、batch size 和测量方式。
- **引用规范**：如果引用第三方项目、论文、模型或数据集，请保留原始链接和许可证信息。
- **高级特性说明**：涉及 MoE、LoRA、Sparse SAHI、CW-NMS 等高级特性时，请引用相关 Wiki 页面或论文章节。
- **文档结构**：
  - `docs/` 目录维护 MkDocs 构建的文档站点。修改后请运行 `mkdocs serve` 本地预览。
  - `wiki/` 目录维护 GitHub Wiki 的 Markdown 文件。修改后同步到 GitHub Wiki。
  - `examples/` 目录包含第三方推理和部署示例。新增示例时请附带 `README.md` 说明。
  - `lora_examples/` 目录包含 LoRA 微调示例。新增示例时请说明数据集、rank、alpha 和训练命令。
- **Hugging Face / Colab**：维护 Hugging Face Spaces demo 和 Colab notebook 时，请确保依赖版本与项目 `requirements.txt` 一致。

---

## 模型数据和权重

请只提交你有权发布的模型、权重、配置和数据说明。大型文件应通过 Release、对象存储、模型仓库（如 Hugging Face）或文档链接分发，不应直接提交到 Git 历史。

如果贡献新的 benchmark 或模型结果，请说明：

- 训练数据、验证数据和评估指标（如 MS COCO AP、AP50、AP75）。
- 训练命令、配置文件、随机种子和关键超参数（如专家数量、top-k、负载均衡系数、LoRA rank 和 alpha）。
- 硬件环境和推理延迟测量方式（如 GPU 型号、TensorRT 版本、batch size、warm-up 次数）。
- 权重下载地址、哈希值或版本号。

---

## PR 评审标准

审查 Pull Requests 是另一种宝贵的贡献方式。当审查 PR 时，请重点关注：

- **问题清晰度**：改动是否解决了清晰的问题？
- **向后兼容**：是否保持向后兼容和可维护性？
- **测试覆盖**：是否有足够测试、文档或复现说明？
- **性能影响**：是否对训练、推理、导出、部署或现有模型结果产生负面影响？（尤其关注 MoE 路由开销、内存占用和延迟）
- **CI 状态**：确认所有 GitHub Actions CI 测试是否通过。
- **建设性反馈**：提供具体、清晰的反馈，认可作者的工作以维持积极的协作氛围。
- **许可证合规**：是否符合许可证、版权和数据合规要求？

---

## 许可证和版权

YOLO-Master 使用 [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE)。提交贡献即表示你同意你的贡献在本项目许可证下发布，并确认你有权提交相关代码、文档、配置、模型或数据说明。

本仓库 `LICENSE` 文件包含 Tencent 对 YOLO-Master 及 Tencent modifications 的版权声明。请不要引入与 AGPL-3.0 不兼容的代码、数据或依赖；如果你复用了第三方内容，请在代码注释、文档或 PR 描述中明确来源和许可证。

本项目基于优秀的 [Ultralytics](https://github.com/ultralytics/ultralytics) 开源框架演进而来。贡献时请保留必要的上游署名、许可证和版权信息。

---

## 基于 YOLO-Master 的开源项目

在你的项目中使用 YOLO-Master 的代码或模型？AGPL-3.0 许可证要求你的整个衍生作品也必须以 AGPL-3.0 开源。这确保了基于开源基础构建的改进和更大的项目始终保持开放。

### 为什么 AGPL-3.0 合规很重要

- **保持软件开放**：确保改进和衍生作品惠及整个社区。
- **法律要求**：使用 AGPL-3.0 许可代码即表示你的项目受其条款约束。
- **促进协作**：鼓励分享和透明。

### 如何遵守 AGPL-3.0

遵守意味着将你的项目**完整对应的源代码**以 AGPL-3.0 许可证公开。

1. **选择起点**：
   - **Fork YOLO-Master**：如果你紧密基于 YOLO-Master 构建，直接 Fork [Tencent/YOLO-Master](https://github.com/Tencent/YOLO-Master)。
   - **使用模板**：从 Ultralytics 模板仓库或 YOLO-Master 结构出发，建立清晰的模块化集成。

2. **许可证声明**：
   - 添加 `LICENSE` 文件，包含完整的 [AGPL-3.0 许可证](https://opensource.org/license/agpl-v3) 文本。
   - 在每个源文件顶部添加许可证声明。

3. **发布源代码**：
   - 将你的**整个项目源代码**公开（例如在 GitHub 上）。这包括：
     - 包含 YOLO-Master 模型或代码的完整应用程序或系统。
     - 对原始 YOLO-Master 代码的任何修改。
     - 训练、验证、推理脚本。
     - 修改或微调后的**模型权重**（如 `.pt`、`.pth`、`.onnx`、`.engine` 等格式）。
     - **配置文件**（`*.yaml` / `*.json`）、环境配置（`requirements.txt`、`Dockerfile`）。
     - 如果是[网页应用](https://en.wikipedia.org/wiki/Web_application)，包含前后端代码。
     - 你修改过的任何第三方库。
     - 运行/重新训练所需的**训练数据**，如果允许再分发。

4. **清晰文档化**：
   - 更新 `README.md`，说明项目以 AGPL-3.0 许可。
   - 包含清晰的设置、构建和运行说明。
   - 适当引用 YOLO-Master，链接回[原始仓库](https://github.com/Tencent/YOLO-Master)。示例：
     ```markdown
     This project utilizes code from [YOLO-Master](https://github.com/Tencent/YOLO-Master), licensed under AGPL-3.0.
     ```

### 示例仓库结构

```
my-yolo-master-project/
│
├── LICENSE                    # Full AGPL-3.0 license text
├── README.md                  # Project description, setup, usage, license & attribution
├── pyproject.toml             # Dependencies (or requirements.txt)
├── scripts/                   # Training/inference scripts
│   └── train.py
├── src/                       # Your project's source code
│   ├── __init__.py
│   ├── data_loader.py
│   └── model_wrapper.py       # Code interacting with YOLO-Master
├── tests/                     # Unit/integration tests
├── configs/                   # YAML/JSON config files
├── docker/                    # Dockerfiles, if used
│   └── Dockerfile
└── .github/                   # GitHub specific files (e.g., CI workflows)
    └── workflows/
        └── ci.yml
```

遵循这些指南，你将确保 AGPL-3.0 合规，支持使 YOLO-Master 等强大工具成为可能的开源生态系统。

---

## 安全问题

如果你发现安全漏洞，请避免在公开 Issue 中披露可直接利用的细节。优先使用 GitHub Security Advisory（如果仓库已启用）或通过维护者认可的私密渠道联系项目维护者；如果必须公开提交 Issue，请只描述影响范围和联系请求。

---

## 常见问题

### 为什么我应该向 YOLO-Master 贡献？

向 YOLO-Master 贡献不仅能改进软件本身，使其对社区更健壮、功能更丰富，还能让你与计算机视觉领域的优秀开发者协作。YOLO-Master 已入选 [腾讯犀牛鸟开源计划](https://opensource.tencent.com/summer-of-code/)，贡献者有机会获得导师指导、资源支持和社区曝光。贡献可以包括代码增强、Bug 修复、文档改进、MoE 路由优化、LoRA 微调方案等。关于如何开始，请参阅 [Pull Request 流程](#pull-request-流程) 部分。

### 如何签署 YOLO-Master 的 CLA？

提交 PR 后，CLA 机器人会引导你完成签署流程。在 PR 中评论以下语句即可：

```text
I have read the CLA Document and I sign the CLA
```

更多信息请参阅 [CLA 签署](#cla-签署) 部分。

### 什么是 Google-style docstrings，为什么 YOLO-Master 要求使用？

Google-style docstrings 为函数和类提供清晰、简洁的文档，提高代码可读性和可维护性。它们概述函数的目的、参数和返回值，遵循特定的格式规则。在 YOLO-Master 中，遵循 Google-style docstrings 确保你的新增内容被良好记录并易于理解。示例和指南请参阅 [代码规范](#代码规范) 部分。

### 如何确保我的改动通过 GitHub Actions CI 测试？

在 PR 被合并之前，必须通过所有 GitHub Actions CI 测试。这些测试包括 linting、单元测试和其他质量检查。请查看 CI 输出并修复任何报错。你可以在本地先运行：

```bash
python -m pytest tests
ruff check ultralytics tests
```

关于 CI 流程和故障排除的详细信息，请参阅 [测试与验证](#测试与验证) 部分。

### 如何报告 YOLO-Master 中的 Bug？

报告 Bug 时，请提供一个清晰、简洁的**最小可复现示例**（Minimum Reproducible Example）。这有助于开发者快速识别和修复问题。请确保你的示例最小但足以复现问题。更详细的步骤请参阅 [报告 Bug](#报告-bug) 部分。

### 如果我在自己的项目中使用 YOLO-Master，AGPL-3.0 意味着什么？

如果你在自己的项目中使用 YOLO-Master 的代码或模型（以 AGPL-3.0 许可），AGPL-3.0 要求你的整个项目（衍生作品）也必须以 AGPL-3.0 许可，并且其完整源代码必须公开。这确保了软件的开源性质在其衍生作品中得以保留。如果你无法满足这些要求，请考虑其他方案。详情请参阅 [基于 YOLO-Master 的开源项目](#基于-yolo-master-的开源项目) 部分。

### YOLO-Master 与 Ultralytics YOLO 有什么关系？

YOLO-Master 基于优秀的 [Ultralytics](https://github.com/ultralytics/ultralytics) 开源框架演进而来，并在其上深度融合了 **ES-MoE (Efficient Sparse Mixture-of-Experts)**、**Dynamic Routing**、**LoRA 高效微调**、**Sparse SAHI**、**CW-NMS**、**MoA (Mixture-of-Attention)** 和 **MoT (Mixture-of-Transformers)** 等创新。我们保留了 Ultralytics 的易用性和工程化标准，同时扩展了 YOLO 架构在动态计算、参数高效微调和稀疏推理方面的能力。贡献时请尊重上游的署名和许可证要求。

### 我可以贡献 MoE 相关的新功能吗？

当然可以。MoE 是 YOLO-Master 的核心创新方向。我们欢迎以下类型的贡献：
- 新的专家架构（如不同卷积类型、注意力机制的专家）
- 改进的路由策略（如基于内容的路由、分层路由）
- 负载均衡损失的改进（如新的辅助损失项）
- 专家剪枝和压缩策略
- 不同硬件平台（GPU / CPU / NPU / Jetson）上的 MoE 部署优化

贡献前请先阅读 [MoE 模块贡献指南](#moe-模块贡献指南) 和项目 Wiki 中的 MoE 原理说明，确保你的改动与现有架构兼容。

### 我应该如何验证 MoE 改动？

MoE 改动的验证应至少包含：
1. 单元测试：`python -m pytest tests/test_moe.py`
2. 辅助损失测试：`python -m pytest tests/test_mixture_aux_loss.py`
3. 修复回归测试：`python -m pytest tests/test_mixture_fixes.py`
4. 专家诊断：`python -c "from ultralytics import YOLO; YOLO('yolo-master-n.yaml').diagnose_model()"`
5. 导出兼容性：`python -m pytest tests/test_exports.py`（如果改动影响路由网络）
6. 训练 smoke test：在 `coco8.yaml` 上运行至少 3 个 epoch 的训练，确认 `moe_loss` 正常收敛。

---

感谢你帮助 YOLO-Master 变得更可靠、更易用、更开放。我们期待你的 Issue 和 Pull Request！🚀🌟
