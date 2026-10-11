<!-- 1）把fused_residual_rmsnorm_backward.cu fused_residual_rmsnorm_forward.cu rmsnorm_backward.cu 均仿照rmsnorm_backward.cu，添加混合精度的算子，然后修改main函数
2）修改readme.md,更新到最新进度，把它改为一份详细的项目，介绍我如何从数学公式，到cpp实现，再到各种优化。同时介绍如何使用nsys和ncu进行优化
3）把result里面的文件命名更加规范，画图脚本修改代码使得画图风格统一 -->

<!-- 用ncu先测试batch-size = 256的情况下的各个算子的瓶颈
最终再写一版在256下的最优 -->
``` 
// 指令
ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel1" --launch-skip 6006 --launch-count 1 ./dev/rmsnorm_forward 1 
ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel2" --launch-skip 6006 --launch-count 1 ./dev/rmsnorm_forward 2 
ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel3" --launch-skip 6006 --launch-count 1 ./dev/rmsnorm_forward 3 
ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel4" --launch-skip 6006 --launch-count 1 ./dev/rmsnorm_forward 4 
ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel5" --launch-skip 6006 --launch-count 1 ./dev/rmsnorm_forward 5 
ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel6" --launch-skip 6006 --launch-count 1 ./dev/rmsnorm_forward 6 

ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel7" --launch-skip 6006 --launch-count 1 ./dev/rmsnorm_forward 7 
ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel7" --launch-skip 18012 --launch-count 1 ./dev/rmsnorm_forward 7 
ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel7" --launch-skip 30018 --launch-count 1 ./dev/rmsnorm_forward 7 

ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel8" --launch-skip 6006 --launch-count 1 ./dev/rmsnorm_forward 8 
ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel8" --launch-skip 18012 --launch-count 1 ./dev/rmsnorm_forward 8 
ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel8" --launch-skip 30018 --launch-count 1 ./dev/rmsnorm_forward 8 

ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel9" --launch-skip 6006 --launch-count 1 ./dev/rmsnorm_forward 9 
ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel9" --launch-skip 18012 --launch-count 1 ./dev/rmsnorm_forward 9 
ncu -f -o result/b256 --set full --kernel-name "rmsnorm_forward_kernel9" --launch-skip 30018 --launch-count 1 ./dev/rmsnorm_forward 9 

  --page arg (=details)                 Select report page to output:
                                          details: sections and rules
                                          raw: all collected metrics
                                          source: source code
                                          session: session and device attributes

ncu -i result/b256.ncu-rep --page details

ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel1_b256_ncu.txt
ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel2_b256_ncu.txt
ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel3_b256_ncu.txt
ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel4_b256_ncu.txt
ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel5_b256_ncu.txt
ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel6_b256_ncu.txt

ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel7fp32_b256_ncu.txt
ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel7fp16_b256_ncu.txt
ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel7bf16_b256_ncu.txt

ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel8fp32_b256_ncu.txt
ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel8fp16_b256_ncu.txt
ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel8bf16_b256_ncu.txt

ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel9fp32_b256_ncu.txt
ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel9fp16_b256_ncu.txt
ncu -i result/b256.ncu-rep --page details > result/20260901/rmsnorm_forward_kernel9bf16_b256_ncu.txt
```

测试在tonleiwen@100.102.239.35 服务器进行，可以把tonleiwen@100.102.239.35:d:/shared-tlw-win/my_cuda_kernels作为中转
在 tonleiwen@100.102.239.35 开启wsl，在home/tongleiwen目录下，有my_cuda_kernels文件夹，在该目录下进行测试

中转的意思是先把本地文件传输到该文件夹，然后打开wsl，从该文件夹传输到wsl里面的my_cuda_kernels下
需要在wsl下的my_cuda_kernels 先创建result/20260902文件夹

本地python环境：conda activate py313

<!-- 我给rmsnorm_forward.cu 新添加了一个kernel9 -->
<!-- 要做什么：1）检查本地rmsnorm_forward.cu 是否正确传输到wsl 2）用nvcc -O3 -std=c++17 -arch=sm_86 --use_fast_math rmsnorm_forward.cu -o rmsnorm_forward重新编译该文件，编译成功后告诉我 -->

<!-- 要做什么：
1）wsl下：把我误放在result/20260901/下的rmsnorm_forward_kernel9fp32_b256_ncu.txt rmsnorm_forward_kernel9fp16_b256_ncu.txt rmsnorm_forward_kernel9bf16_b256_ncu.txt 移动到20260902文件夹下
2）在本地result创建20260902文件夹，把 rmsnorm_forward_kernel9fp32_b256_ncu.txt rmsnorm_forward_kernel9fp16_b256_ncu.txt rmsnorm_forward_kernel9bf16_b256_ncu.txt 和 rmsnorm_forward.txt同步到本地
3）结合20260831的rmsnorm_forward.txt和今天的rmsnorm_forward.txt,在20260902文件夹下生成rmsnorm_forward.png
4)仿照20260901/result/analysis，先复制到20260902,然后补充kernel9的分析内容，生成一份新的analysis -->

<!-- 要做什么：
1）将本地20260902文件夹下新建4个文件夹rmsnorm_forward, rmsnorm_backward,fused_residual_rmsnorm_forward, fused_residual_rmsnorm_backward
2) 将现在本地20260902文件夹下的文件，属于rmsnorm_forward的，全部放到该文件夹下
3) 把wsl下的20260902文件夹也按照1），2）修改
4) 对rmsnorm_backward,fused_residual_rmsnorm_forward, fused_residual_rmsnorm_backward 3个算子均执行如下操作：
   1) 先仿照// 指令,将算子需要的指令写在guide.md(追加在后面)
   2) 在wsl服务器上执行指令，在20260902文件夹对应文件夹下生成数据txt文件
   3) 将数据同步回本地
5）给每个算子 生成一个analysis 以及画图 -->

## 20260902：其余三个算子的 benchmark 与 block256 NCU 指令

以下命令均在 WSL 的 `/home/tongleiwen/my_cuda_kernels` 根目录执行。前向程序每个 block
benchmark 2000 次，block256 对应 `launch-skip=6006`；反向程序每个 block benchmark
100 次，block256 对应 `launch-skip=306`。混合精度程序按 FP32 → FP16 → BF16 依次运行，
所以前向三段 skip 为 `6006/18012/30018`，反向三段为 `306/912/1518`。

```bash
mkdir -p result/20260902/{rmsnorm_forward,rmsnorm_backward,fused_residual_rmsnorm_forward,fused_residual_rmsnorm_backward}

# benchmark 文本
./dev/rmsnorm_backward 1 > result/20260902/rmsnorm_backward/rmsnorm_backward.txt
./dev/rmsnorm_backward 2 >> result/20260902/rmsnorm_backward/rmsnorm_backward.txt
./dev/rmsnorm_backward 3 >> result/20260902/rmsnorm_backward/rmsnorm_backward.txt
./dev/rmsnorm_backward 4 >> result/20260902/rmsnorm_backward/rmsnorm_backward.txt
./dev/rmsnorm_backward 5 >> result/20260902/rmsnorm_backward/rmsnorm_backward.txt
./dev/rmsnorm_backward 6 >> result/20260902/rmsnorm_backward/rmsnorm_backward.txt
./dev/rmsnorm_backward 7 >> result/20260902/rmsnorm_backward/rmsnorm_backward.txt
./dev/rmsnorm_backward 8 >> result/20260902/rmsnorm_backward/rmsnorm_backward.txt

./dev/fused_residual_rmsnorm_forward 1 > result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward.txt
./dev/fused_residual_rmsnorm_forward 2 >> result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward.txt
./dev/fused_residual_rmsnorm_forward 3 >> result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward.txt
./dev/fused_residual_rmsnorm_forward 4 >> result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward.txt
./dev/fused_residual_rmsnorm_forward 5 >> result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward.txt
./dev/fused_residual_rmsnorm_forward 6 >> result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward.txt
./dev/fused_residual_rmsnorm_forward 7 >> result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward.txt
./dev/fused_residual_rmsnorm_forward 8 >> result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward.txt
./dev/fused_residual_rmsnorm_forward 9 >> result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward.txt

./dev/fused_residual_rmsnorm_backward 1 > result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward.txt
./dev/fused_residual_rmsnorm_backward 2 >> result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward.txt
./dev/fused_residual_rmsnorm_backward 3 >> result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward.txt
./dev/fused_residual_rmsnorm_backward 4 >> result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward.txt
./dev/fused_residual_rmsnorm_backward 5 >> result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward.txt
./dev/fused_residual_rmsnorm_backward 6 >> result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward.txt
./dev/fused_residual_rmsnorm_backward 7 >> result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward.txt
./dev/fused_residual_rmsnorm_backward 8 >> result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward.txt

# 统一采集并导出 details 文本；第 1 个参数不要带 .ncu-rep 后缀
profile_ncu() {
  out="$1"
  kernel="$2"
  skip="$3"
  binary="$4"
  version="$5"
  ncu -f -o "$out" --set full --kernel-name "$kernel" \
      --launch-skip "$skip" --launch-count 1 "./dev/$binary" "$version"
  ncu -i "${out}.ncu-rep" --page details > "${out}_ncu.txt"
}

# RMSNorm backward：kernel1~5 FP32，kernel6~7 为 FP32/FP16/BF16
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel1_b256 rmsnorm_backward_kernel1 306 rmsnorm_backward 1
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel2_b256 rmsnorm_backward_kernel2 306 rmsnorm_backward 2
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel3_b256 rmsnorm_backward_kernel3 306 rmsnorm_backward 3
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel4_b256 rmsnorm_backward_kernel4 306 rmsnorm_backward 4
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel5_b256 rmsnorm_backward_kernel5 306 rmsnorm_backward 5
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel6fp32_b256 rmsnorm_backward_kernel6 306 rmsnorm_backward 6
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel6fp16_b256 rmsnorm_backward_kernel6 912 rmsnorm_backward 6
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel6bf16_b256 rmsnorm_backward_kernel6 1518 rmsnorm_backward 6
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel7fp32_b256 rmsnorm_backward_kernel7 306 rmsnorm_backward 7
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel7fp16_b256 rmsnorm_backward_kernel7 912 rmsnorm_backward 7
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel7bf16_b256 rmsnorm_backward_kernel7 1518 rmsnorm_backward 7

# kernel8 是 FP32/FP16/BF16 三个非模板专用 kernel。使用精确 kernel 名称后每种
# dtype 独立计数；block1024 因资源上限回退到 kernel6，所以 block256 skip=305。
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel8fp32_b256 rmsnorm_backward_kernel8_fp32 305 rmsnorm_backward 8
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel8fp16_b256 rmsnorm_backward_kernel8_fp16 305 rmsnorm_backward 8
profile_ncu result/20260902/rmsnorm_backward/rmsnorm_backward_kernel8bf16_b256 rmsnorm_backward_kernel8_bf16 305 rmsnorm_backward 8

# Fused Residual RMSNorm forward：kernel1~6 FP32，kernel7~8 为 FP32/FP16/BF16
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel1_b256 fused_residual_rmsnorm_forward_kernel1 6006 fused_residual_rmsnorm_forward 1
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel2_b256 fused_residual_rmsnorm_forward_kernel2 6006 fused_residual_rmsnorm_forward 2
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel3_b256 fused_residual_rmsnorm_forward_kernel3 6006 fused_residual_rmsnorm_forward 3
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel4_b256 fused_residual_rmsnorm_forward_kernel4 6006 fused_residual_rmsnorm_forward 4
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel5_b256 fused_residual_rmsnorm_forward_kernel5 6006 fused_residual_rmsnorm_forward 5
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel6_b256 fused_residual_rmsnorm_forward_kernel6 6006 fused_residual_rmsnorm_forward 6
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel7fp32_b256 fused_residual_rmsnorm_forward_kernel7 6006 fused_residual_rmsnorm_forward 7
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel7fp16_b256 fused_residual_rmsnorm_forward_kernel7 18012 fused_residual_rmsnorm_forward 7
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel7bf16_b256 fused_residual_rmsnorm_forward_kernel7 30018 fused_residual_rmsnorm_forward 7
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel8fp32_b256 fused_residual_rmsnorm_forward_kernel8 6006 fused_residual_rmsnorm_forward 8
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel8fp16_b256 fused_residual_rmsnorm_forward_kernel8 18012 fused_residual_rmsnorm_forward 8
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel8bf16_b256 fused_residual_rmsnorm_forward_kernel8 30018 fused_residual_rmsnorm_forward 8

# kernel9 是 FP32/FP16/BF16 三个非模板专用 kernel
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel9fp32_b256 fused_residual_rmsnorm_forward_kernel9_fp32 6006 fused_residual_rmsnorm_forward 9
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel9fp16_b256 fused_residual_rmsnorm_forward_kernel9_fp16 6006 fused_residual_rmsnorm_forward 9
profile_ncu result/20260902/fused_residual_rmsnorm_forward/fused_residual_rmsnorm_forward_kernel9bf16_b256 fused_residual_rmsnorm_forward_kernel9_bf16 6006 fused_residual_rmsnorm_forward 9

# Fused Residual RMSNorm backward：kernel1~5 FP32，kernel6~7 为 FP32/FP16/BF16
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel1_b256 fused_residual_rmsnorm_backward_kernel1 306 fused_residual_rmsnorm_backward 1
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel2_b256 fused_residual_rmsnorm_backward_kernel2 306 fused_residual_rmsnorm_backward 2
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel3_b256 fused_residual_rmsnorm_backward_kernel3 306 fused_residual_rmsnorm_backward 3
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel4_b256 fused_residual_rmsnorm_backward_kernel4 306 fused_residual_rmsnorm_backward 4
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel5_b256 fused_residual_rmsnorm_backward_kernel5 306 fused_residual_rmsnorm_backward 5
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel6fp32_b256 fused_residual_rmsnorm_backward_kernel6 306 fused_residual_rmsnorm_backward 6
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel6fp16_b256 fused_residual_rmsnorm_backward_kernel6 912 fused_residual_rmsnorm_backward 6
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel6bf16_b256 fused_residual_rmsnorm_backward_kernel6 1518 fused_residual_rmsnorm_backward 6
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel7fp32_b256 fused_residual_rmsnorm_backward_kernel7 306 fused_residual_rmsnorm_backward 7
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel7fp16_b256 fused_residual_rmsnorm_backward_kernel7 912 fused_residual_rmsnorm_backward 7
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel7bf16_b256 fused_residual_rmsnorm_backward_kernel7 1518 fused_residual_rmsnorm_backward 7

# kernel8 是 FP32/FP16/BF16 三个非模板专用 kernel
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel8fp32_b256 fused_residual_rmsnorm_backward_kernel8_fp32 305 fused_residual_rmsnorm_backward 8
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel8fp16_b256 fused_residual_rmsnorm_backward_kernel8_fp16 305 fused_residual_rmsnorm_backward 8
profile_ncu result/20260902/fused_residual_rmsnorm_backward/fused_residual_rmsnorm_backward_kernel8bf16_b256 fused_residual_rmsnorm_backward_kernel8_bf16 305 fused_residual_rmsnorm_backward 8
```
<!-- 要做什么：
1）为rmsnorm_backward, fused_resiual_rmsnorm_forward, fused_resiual_rmsnorm_backward原代码先保持不变，各添加3个优化算子，3各算子分别是在fp32,fp16,bf16精度下的优化版本，不要写成模板，根据现有的数据尽可能优化。
2）同步 rmsnorm_backward, fused_resiual_rmsnorm_forward, fused_resiual_rmsnorm_backward新代码到wsl对应目录
3）把新算子的对应指令补充到## 20260902：其余三个算子的 benchmark 与 block256 NCU 指令
4）在wsl重新编译，追加测试结果txt和生成对应ncu报告txt，然后同步回本地
5）更新图像和analysis -->

1. 我把result放进了gitignore，因为里面有原始数据，不要上传原始数据
2. 新建一个文件夹script，把运行脚本放进去
3. dev下面只放代码
4. 新建doc，把分析的md放在这里，ignore
5. 重新在analysis放每个算子的分析结果报告，可以先在result画好图，画图风格统一，一共10个，分别是rmsnorm forward，backward，fused forward，backward，tilelang 版的rmsnorm forward，backward，fused forward backward，gemm，grouped gemm
6. 把dev下的cuda算子，先新建一个文件夹cuda，和tilelang同级，把cuda算子放进去
7. 更新readme，readme不需要写详细结果，指向anlysis即可
8. 更新git，并上传远程仓库
9. 最终上传只有dev anlysis gitignore readme

## 20261005 执行记录

CUDA源码已整理到dev/cuda，与dev/tilelang同级；活跃脚本在script，详细文档在doc，公开分析在analysis，统一图表在figures。result/script/doc仅保留本地，原始数据取消Git跟踪。README提供十份分析入口。TileLang使用SCVideo的GCC12.4与tilelang-3032的.venv；两个backward的BLOCK_N128配置因长时间编译未完成，测试只涵盖BLOCK_N1/32和各自原始splitc候选（RMSNorm backward为6组，其他为7组）。


## 20261007：kernel9 动态 M

保留原 kernel8，新增 kernel9。三组固定 (K,N) 为 (768,2432)/(2432,768)/(1024,1024)，FP16/BF16、E<=16，GPU counts 驱动元数据与任务调度。单 Rank 算子验证和同轮 CUTLASS 对照已完成；正式 PyTorch 注册和模型接入是下一阶段。

服务器工作目录：/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03。使用 TileLang venv 的 Python 与 SCVideo 的 GCC12。

```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
python source/script/benchmark_kernel9.py --library results/R13_release_build/kernel9_bridge.so --validate-only --output results/REPRO_validation
python source/script/benchmark_kernel9.py --library results/R13_release_build/kernel9_bridge.so --benchmark-only --default-policy --cutlass-selection results/R07_tuning_fixed/benchmark.json --output results/REPRO_default_benchmark
CHECK_BINARY=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03/results/R13_release_build/check_kernel9 SANITIZER_OUT=/data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03/results/REPRO_sanitizer BENCH_GPU=1 bash source/script/sanitize_kernel9.sh
python source/script/plot_kernel9.py --results results/R14_default_benchmark/benchmark.json --output results/figures/grouped_gemm_kernel9.png
```

使用新的 REPRO 输出目录；正式安全检查为 22 组通过。最终性能报告入口为 analysis/grouped_gemm_kernel9.md。R18 NCU 尝试因 GPU 性能计数器权限不足未采集数据。


## 20261007：kernel9 PyTorch 接口

已新增 CUDA dispatcher 绑定、Python 包装、显式工作区与专家权重缓存。GEMM 核心复用原 kernel9。R05 扩展在当前 torch 2.9.0+cu128 / C++11 ABI=true 构建；R06 完整接口、R07 同轮测量、R08 memcheck 完成。

在 Server-Internet 的 /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04 执行：

```bash
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export CXX=/data/share/home/user4/.conda/envs/SCVideo/bin/x86_64-conda-linux-gnu-c++
export CUDA_HOME=/usr/local/cuda
export OMP_NUM_THREADS=1
python build_api.py --source source --output results/REPRO_api_build
CUDA_VISIBLE_DEVICES=1,2 python verify_api.py --source source --library results/REPRO_api_build/my_cuda_kernel9.so --output results/REPRO_api_validation
CUDA_VISIBLE_DEVICES=1 python benchmark_api.py --source source --library results/REPRO_api_build/my_cuda_kernel9.so --native /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage03/results/R13_release_build/kernel9_bridge.so --output results/REPRO_api_benchmark
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 97 python check_api_sanitizer.py --source source --library results/REPRO_api_build/my_cuda_kernel9.so
```

目录需使用新的输出轮次。接口分析入口为 analysis/grouped_gemm_api.md；下一阶段完整专家 MLP 回放。


## 20261007：完整专家 MLP 回放

已增加 GroupedExpertMLP：三次 kernel9 加原 BF16 SiLU/乘法边界，复用 GEMM 与激活缓冲，空输入跳过计算。100 个真实回放通过 FP32 舍入链参考；对原 BF16 输出旧阈值仍有 3 例、72 个差异，模型级接入需继续追踪，不能默认判为完全兼容。完整 MLP 的同轮原实现/CUTLASS 对照与图见 analysis/grouped_expert_mlp.md。

在 Server-Internet 的 /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage05 执行：

```bash
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export OMP_NUM_THREADS=1
K9_RUN=REPRO_release_audit bash run_replay.sh --validate-only --audit-errors
K9_RUN=REPRO_full_mlp_benchmark bash run_replay.sh --benchmark-only --audit-errors --cutlass-root /data/user4/tongleiwen/moe-open-baselines
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 97 python check_mlp_sanitizer.py --source source --library /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage04/results/R05_dispatch_build/my_cuda_kernel9.so --checkpoint /data/user4/tongleiwen/minimind/out/pretrain_ep_768_moe.pth --output results/REPRO_memory/summary.json > results/REPRO_memory.console.log 2>&1
python plot_mlp.py --results results/R06_final_benchmark/benchmark.json --output results/figures/grouped_expert_mlp.png
```

输出目录使用新名称。R08 最终源码 100 回放与 R03/R05 数值结果完全一致；R09 5 组完整链、4 组 Graph counts 及生命周期检查通过，memcheck 零错误。Graph/eager 均单独呈现，工作区准备与 EP 通信留到下一阶段测量。


## 20261007：MiniMind-EP 阶段06/07

已完成双卡可切换后端、模型误差传播、训练回退和模型/EP/连续Decode测量。GPU1/2首轮遭其它任务占用OOM，后续验证与正式性能在空闲GPU4/5、CPU28/29进行，默认original。全量kernel9/hybrid不满足模型数值门槛；last_layer在测试范围通过，但整体收益约0.x%，增长KV基本持平或略慢。分析见analysis/minimind_ep_integration.md。

```bash
cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage06
source /data/user4/tongleiwen/tilelang-3032/.venv/bin/activate
export BENCH_GPUS=4,5
export BENCH_CPU_BASE=28
K9_RUN=REPRO_ep_validation bash run_validation.sh

cd /data/user4/tongleiwen/minimind-ep-grouped-baseline-20261007/stage07
K9_RUN=REPRO_model bash run_measure.sh
K9_RUN=REPRO_profile bash run_measure.sh --profile-only
K9_RUN=REPRO_repeat bash run_measure.sh --workloads prefill_small,decode_single,decode_batch --no-ep --no-growing
python analyze_model.py --input results/REPRO_model --profile results/REPRO_profile --output results/REPRO_analysis.json
python analyze_model.py --input results/REPRO_repeat --output results/REPRO_repeat_analysis.json
python plot_model.py --results results/REPRO_analysis.json --output results/figures/REPRO_minimind_ep_model.png
```

各任务按顺序执行，性能测量期间不运行profiler或数值验证。R03阶段06完整审计、R04快速/审计60组位级一致、R05批量16连续Decode检查，以及阶段07主测/诊断/独立复测均完成。详细快路径与B16检查指令见doc/06_EP接入验证.md。
