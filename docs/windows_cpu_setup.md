# Windows CPU 推理配置指南 🪟

本文档提供在 **Windows 11 + Python 3.9 + CPU Only** 环境下配置和运行 YOLO-Master 的完整步骤。

---

## 环境信息

| 项目 | 详情 |
|------|------|
| 操作系统 | Windows 11 版本 10.0.26200.8655 |
| CPU | Intel(R) Core(TM) Ultra 5 225H |
| Python 版本 | Python 3.9.7 |
| 运行模式 | CPU Only（无 CUDA 显卡） |
| YOLO-Master 版本 | main 分支 |

---

## 1. 克隆仓库并创建虚拟环境

```bash
git clone https://github.com/Tencent/YOLO-Master.git
cd YOLO-Master
python -m venv venv

# 激活虚拟环境 (Windows PowerShell)
.\venv\Scripts\activate
```

> 如果使用 Command Prompt (cmd.exe)，激活命令为 `venv\Scripts\activate.bat`。

---

## 2. 安装 PyTorch CPU 版本（关键步骤）

这是整个配置过程中最容易出问题的环节。在 Windows + CPU 模式下，如果直接执行 `pip install -r requirements.txt`，pip 会默认拉取 PyTorch 的 CUDA 编译版本。由于没有 CUDA 硬件，运行时会产生以下错误：

### 常见错误

```
RuntimeError: Could not find CUDA drivers
ImportError: DLL load failed while importing _C: 找不到指定的模块。
```

### 正确做法

**先手动安装 PyTorch CPU 版本，再安装其余依赖：**

```bash
# 安装 PyTorch CPU 版本（适配 Python 3.9 + Windows）
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

> ⚠️ **版本兼容性说明**
>
> YOLO-Master 的 `pyproject.toml` 已明确排除 PyTorch 2.4.0 在 Windows 上的使用：
> ```
> torch>=1.8.0,!=2.4.0; sys_platform == 'win32'
> ```
> 这是因为 PyTorch 2.4.0 在 Windows CPU 环境下存在 DLL 加载兼容性问题（参见 [ultralytics/ultralytics#15049](https://github.com/ultralytics/ultralytics/issues/15049)）。
>
> **推荐版本组合：**
> - PyTorch 2.0.1 + TorchVision 0.15.2 ✅
> - PyTorch 2.1.2 + TorchVision 0.16.2 ✅
> - PyTorch 2.4.0 ❌（Windows 下被排除）

---

## 3. 安装其余依赖

```bash
pip install -r requirements.txt
```

如果安装过程中遇到 `peft` 版本冲突（约束为 `peft>=0.18.0,<0.20.0`），可以暂时跳过 —— peft 仅在训练 LoRA 模型时需要，CPU 推理完全不受影响。可手动安装核心依赖：

```bash
pip install numpy>=1.23.0 matplotlib>=3.3.0 opencv-python>=4.6.0 \
    pillow>=7.1.2 pyyaml>=5.3.1 requests>=2.23.0 scipy>=1.4.1 \
    psutil>=5.8.0 polars>=0.20.0 ultralytics-thop>=2.0.18 \
    tqdm>=4.64.0 pandas>=1.1.4 seaborn>=0.11.0
```

---

## 4. 验证安装

在 Python 交互环境中运行以下代码确认安装成功：

```python
import torch
print(f"PyTorch version: {torch.__version__}")          # Expected: 2.0.1+cpu
print(f"CUDA available: {torch.cuda.is_available()}")   # Expected: False
print(f"CPU threads: {torch.get_num_threads()}")        # Depends on your CPU

from ultralytics import YOLO
print("Ultralytics YOLO loaded successfully!")
```

**预期输出：**
```
PyTorch version: 2.0.1+cpu
CUDA available: False
CPU threads: 16
Ultralytics YOLO loaded successfully!
```

---

## 5. 运行 CPU 推理

### 5.1 下载预训练模型

```bash
# 从 Model Zoo 下载预训练权重
python scripts/download_weights.py
```

或者手动从 [YOLO-Master Model Zoo](https://github.com/Tencent/YOLO-Master#model-zoo) 下载 `.pt` 文件到项目根目录。

### 5.2 命令行推理

**图像推理：**
```bash
python app.py --source path/to/your/image.jpg --weights yolo-master-n.pt --device cpu
```

**视频推理：**
```bash
python app.py --source path/to/your/video.mp4 --weights yolo-master-n.pt --device cpu
```

### 5.3 Python API 推理

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo-master-n.pt")

# CPU 推理
results = model("path/to/image.jpg", device="cpu")

# 查看结果
for r in results:
    print(f"Detected {len(r.boxes)} objects")
    for box in r.boxes:
        print(f"  Class: {model.names[int(box.cls)]}, "
              f"Confidence: {float(box.conf):.2f}")
    r.show()  # 显示检测结果
    # r.save("output.jpg")  # 保存结果图片
```

---

## 6. Windows 特定注意事项

### 6.1 路径处理
Windows 下的路径分隔符为 `\`，但在 Python 字符串中建议使用 `/` 或原始字符串，避免转义问题：

```python
# ✅ 推荐
source = "D:/images/photo.jpg"
source = r"D:\images\photo.jpg"

# ❌ 避免
source = "D:\images\photo.jpg"  # \i 可能被解释为转义
```

### 6.2 多进程限制
CPU 推理时，建议设置 `workers=0` 以避免 Windows 下 DataLoader 多进程序列化问题：

```python
results = model("image.jpg", device="cpu", workers=0)
```

### 6.3 性能建议
- **模型选择**：Intel Core Ultra 5 225H 的 16 线程足以流畅运行 YOLO-Master-N 和 YOLO-Master-S 等轻量模型。较大模型（M/L/X）在 CPU 上推理速度较慢，建议优先使用 N/S 变体。
- **内存管理**：CPU 推理时内存占用较高，建议关闭不必要的后台应用。
- **批处理**：如需批量处理多张图片，建议逐张推理而非增大 batch size，以避免 CPU 内存瓶颈。

### 6.4 OpenCV 显示问题
如果 `cv2.imshow()` 窗口无响应或卡死，可改用 `results.save()` 直接保存结果图片，然后手动查看：

```python
results = model("image.jpg", device="cpu")
results[0].save("output.jpg")
```

---

## 故障排查

| 错误信息 | 可能原因 | 解决方案 |
|----------|----------|----------|
| `DLL load failed while importing _C` | 安装了 CUDA 版 PyTorch | 卸载后重装 CPU 版本 |
| `No module named 'torch'` | PyTorch 未安装 | 按第 2 步安装 |
| `RuntimeError: CUDA error` | 代码未指定 `device="cpu"` | 显式传入 `device="cpu"` |
| `ImportError: cannot import name 'YOLO'` | ultralytics 未正确安装 | 在项目根目录执行 `pip install -e .` |
| `MemoryError` | 图片过大或模型过大 | 降低图片分辨率或换用更小的模型 |

---

## 参考链接

- [Ultralytics YOLO 官方文档](https://docs.ultralytics.com/)
- [YOLO-Master 论文 (CVPR 2026)](https://github.com/Tencent/YOLO-Master)
- [PyTorch CPU 版本安装指南](https://pytorch.org/get-started/locally/)
- [Windows PyTorch 2.4.0 兼容性问题](https://github.com/ultralytics/ultralytics/issues/15049)
