# P1 实现的 CPU 测试记录

本组测试对应提交 `5f744437bc7b1620e8d547fbd0c608e80d2ea760`，运行时受 Git 跟踪的源码与该提交一致。四份测试文件共 **69 项通过、0 项失败、0 项错误、0 项跳过**。

| 文件 | 内容 |
| --- | --- |
| [pytest.xml](pytest.xml) | 69 个用例的名称、结果与执行时间 |
| [pytest-output.txt](pytest-output.txt) | 测试输出，包括两项注意力 head 数自动调整提示 |
| [environment.json](environment.json) | 源码版本、CPU 环境、软件版本、配置位置与执行设置 |
| [requirements-test.txt](requirements-test.txt) | 本次测试的主要依赖版本 |

## 运行方式

在仓库根目录、已安装相应依赖的环境中执行：

```bash
YOLO_AUTOINSTALL=false YOLO_OFFLINE=true PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/test_yoloe_p1_text_router.py \
  tests/test_routing_aux_contract.py tests/test_moe_router_boundaries.py \
  tests/test_master_model_configs.py --junitxml=pytest.xml
```

环境为 macOS arm64、Python 3.12.14、PyTorch 2.7.1、pytest 8.4.2，测试使用 CPU。配置目录、字节码和绘图库缓存目录可放在用户自行指定的临时目录。

## 记录处理

本包由同一次测试的原始输出和 JUnit XML生成。处理包括移除终端颜色控制码、将本机绝对路径转为仓库相对路径、移除 JUnit 主机名和时间戳属性；用例名称、计数、结果、耗时和两项 warning 保持原值。原始记录按原样留档，公开副本用于查看该版本的工程验证结果。
