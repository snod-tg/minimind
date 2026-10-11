# 安装、编译与验证命令

## 目的
用同一 Python 环境构建和加载通用算子，不依赖模型阶段目录。

## 执行方案
在项目根目录执行，先激活已有 PyTorch 环境。GPU 构建需要 CUDA PyTorch、nvcc 和匹配的 C++ 编译器；CXX 可指定 SCVideo 的 GCC12。

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m unittest tests.test_package -v
python script/build.py ep --dry-run
python script/build.py grouped_gemm --dry-run
python script/build.py kernel9 --dry-run

# CUDA 机器上执行实际编译
python script/build.py ep
python script/build.py grouped_gemm
python script/build.py kernel9

# EP 张量算子：输出目录每轮独立
python -m tests.ep.validate_functional --library build/ep/current/my_cuda_ep.so --output result/ep_functional.json
python -m tests.ep.validate_configuration --library build/ep/current/my_cuda_ep.so --output result/ep_configuration.json
python -m tests.ep.validate_ops --library build/ep/current/my_cuda_ep.so --output result/ep_ops.json

# 最新 Prefill/Decode API；GPU 测试会创建输出目录
python -m tests.verify_grouped_gemm --library build/grouped_gemm/current/my_cuda_grouped_split.so --output result/grouped_gemm_validation
```

EP 的 --output 是 JSON 文件，先创建其父目录；Grouped GEMM 的 --output 是新目录。工作区、counts、BF16 舍入及支持形状见 [Grouped GEMM API](../../dev/pytorch/README.md)。MY_CUDA_KERNELS_BUILD 可选择其他编译产物根目录，或显式向 load_library 传库路径。

## 结果
本地包安装、CPU 检查和三种编译命令预览已通过；本轮没有进行 GPU 编译/benchmark。

## 调整闭环
构建失败不切换 current；修正工具链后生成新版本。每次性能调整先验证数值，再复测；具体结果写到对应结果文档。
