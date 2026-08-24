# 01 - 环境安装

> 提交日期：2026-08-24
> 锁定版本：main @ e9ac08b
> 对应登记表列：环境安装

## 硬件

| 项 | 值 |
|---|---|
| 操作系统 | Windows 11 (win32) x64 |
| CPU | 12th Gen Intel Core i5-1240P |
| 内存 | ≥ 16 GB |
| GPU | **无 NVIDIA GPU**（按任务规则降级 ONNX Runtime CPU） |

## 软件

| 组件 | 版本 |
|---|---|
| Python | 3.13.5 |
| torch | 2.7.1+cpu |
| ultralytics | 8.4.101（本地 editable） |
| onnx | 1.22.0（opset 19） |
| onnxslim | 0.1.96 |
| onnxruntime | 1.29.0（CPUExecutionProvider） |
| opencv-python | 4.12.0.88 |
| pytest | 9.1.1 |

## 镜像源（本机网络环境限制）

- PyPI：`https://pypi.tuna.tsinghua.edu.cn/simple`
- HuggingFace：`https://hf-mirror.com/`
- GitHub：`https://ghfast.top/`

## 安装步骤

```powershell
# 1. 虚拟环境
cd D:\YOLO_MASTER
python -m venv .venv

# 2. 激活（PowerShell）
.venv\Scripts\Activate.ps1
# 或 Git Bash:
# . .venv/Scripts/activate

# 3. 安装（清华镜像加速）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx onnxruntime
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytest

# 4. 修复 examples 子包笔误（仓库自带，1.19.2 不存在）
# 文件：examples/RTDETR-ONNXRuntime-Python/requirements.txt
# 改：onnxruntime==1.19.2 → onnxruntime>=1.20.0
```

## 验证

```powershell
python -c "import ultralytics, onnxruntime; print('OK', onnxruntime.__version__)"
# 预期：OK 1.29.0
```

## 已知问题

- 仓库 `pyproject.toml` 声明支持 Python 3.13，torch 2.7+ wheel 兼容
- `examples/RTDETR-ONNXRuntime-Python/requirements.txt` 中 `onnxruntime==1.19.2` 是上游笔误（该版本号未发布），已就地修复为 `>=1.20.0`
