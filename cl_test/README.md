# OpenCL 测试程序 - MYD-LR3576开发板

适用于瑞芯微RK3576平台（Mali-G52 MC3 GPU）的OpenCL测试程序。

## 平台信息

| 项目 | 规格 |
|------|------|
| SoC | RK3576 (8nm) |
| GPU | ARM Mali-G52 MC3 @ 1GHz |
| OpenCL | 2.0 |
| GPU算力 | 145 GFLOPS |

## 文件结构

```
rk3576_opencl_test/
├── cl_test.c        # 主程序源码
├── Makefile         # 编译脚本
├── run_test.sh      # 运行脚本
└── README.md        # 说明文档
```

## 快速开始

### 1. 添加执行权限

```bash
chmod +x run_test.sh
```

### 2. 运行测试

```bash
# 快速测试（仅显示GPU信息）
./run_test.sh quick

# 完整基准测试
./run_test.sh bench

# 仅显示GPU信息
./run_test.sh info

# 检查依赖
./run_test.sh check
```

## 手动编译

```bash
# 编译
make

# 调试版本
make debug

# 交叉编译（在PC上编译）
make cross CROSS_COMPILE=aarch64-linux-gnu-

# 清理
make clean
```

## 命令行参数

```
Usage: ./cl_test [options]

Options:
  -h, --help        显示帮助信息
  -i, --info        仅显示平台/设备信息
  -b, --benchmark   运行性能基准测试
  -v, --vector      向量操作基准测试
  -m, --matrix      矩阵乘法基准测试
  -M, --memory      内存带宽基准测试
  -a, --all         运行所有测试（默认）
```

### 示例

```bash
# 显示GPU信息
./cl_test -i

# 运行所有基准测试
./cl_test -b

# 仅运行向量操作测试
./cl_test -v

# 仅运行矩阵乘法测试
./cl_test -m

# 仅运行内存带宽测试
./cl_test -M

# 组合测试
./cl_test -v -m -M
```

## 测试内容

### 1. 平台信息检测
- OpenCL平台信息（厂商、版本、扩展）
- 设备信息（名称、类型、计算单元）
- 内存信息（全局内存、本地内存、缓存）
- 图像支持、浮点精度支持
- 关键扩展支持检查

### 2. 向量操作基准测试
- 向量加法（Vector Addition）
- 向量乘法（Vector Multiplication）
- SAXPY运算（Y = a*X + Y）
- 测量吞吐量（GB/s）

### 3. 矩阵乘法基准测试
- 朴素算法（Naive）
- 分块优化算法（Tiled）
- 测量计算性能（GFLOPS）

### 4. 内存带宽基准测试
- Host → Device 传输带宽
- Device → Host 传输带宽
- Device → Device 复制带宽

## 预期输出示例

```
╔════════════════════════════════════════════════════════════╗
║     OpenCL Test for MYD-LR3576 Development Board           ║
║                   Mali-G52 MC3 GPU                         ║
╚════════════════════════════════════════════════════════════╝

Found 1 OpenCL platform(s)

╔════════════════════════════════════════════════════════╗
║              OpenCL Platform Information               ║
╚════════════════════════════════════════════════════════╝
  Name:       ARM Platform
  Vendor:     ARM
  Version:    OpenCL 2.0
  ...

╔════════════════════════════════════════════════════════╗
║              OpenCL Device Information                 ║
╚════════════════════════════════════════════════════════╝
  Device Name:           Mali-G52
  Vendor:                ARM
  Device Version:        OpenCL 2.0
  Compute Units:         3
  Max Clock Frequency:   1000 MHz
  Global Memory:         1024.00 MB
  ...

╔════════════════════════════════════════════════════════╗
║           Vector Operations Benchmark                  ║
╚════════════════════════════════════════════════════════╝
  Vector Size: 1048576 elements (4.00 MB)

  Vector Addition:      2.156 ms, 5.57 GB/s
  Vector Multiply:      2.134 ms, 5.62 GB/s
  SAXPY:                1.892 ms, 4.23 GB/s

╔════════════════════════════════════════════════════════╗
║          Matrix Multiplication Benchmark               ║
╚════════════════════════════════════════════════════════╝
  Matrix Size: 512x512 x 512x512
  Operations:  0.27 GFLOP

  Naive MatMul:         125.43 ms, 2.15 GFLOPS
  Tiled MatMul:         45.67 ms, 5.91 GFLOPS

╔════════════════════════════════════════════════════════╗
║            Memory Bandwidth Benchmark                  ║
╚════════════════════════════════════════════════════════╝
  Buffer Size: 128.00 MB

  Host -> Device:       3.25 GB/s
  Device -> Host:       3.18 GB/s
  Device -> Device:     8.56 GB/s
```

## 依赖安装

### Debian/Ubuntu

```bash
# OpenCL 运行时（如果Mali驱动未安装）
sudo apt install ocl-icd-opencl-dev

# OpenCL 头文件（开发用）
sudo apt install opencl-headers

# 编译工具
sudo apt install build-essential

# 可选：clinfo工具
sudo apt install clinfo
```

### 米尔RK3576开发板

米尔RK3576开发板通常已预装Mali GPU驱动：

```bash
# 检查Mali库
ls /usr/lib/libmali.so
# 或
ls /usr/lib/aarch64-linux-gnu/libmali.so

# 检查OpenCL ICD
ls /etc/OpenCL/vendors/

# 使用clinfo检查
clinfo
```

## 故障排除

### 找不到OpenCL平台

```bash
# 检查库文件
ldconfig -p | grep OpenCL
ldconfig -p | grep mali

# 检查ICD配置
ls -la /etc/OpenCL/vendors/

# 设置库路径
export LD_LIBRARY_PATH=/usr/lib/mali:$LD_LIBRARY_PATH
```

### 编译错误

```bash
# 安装开发包
sudo apt install build-essential opencl-headers

# 检查头文件
ls /usr/include/CL/
```

### 运行时错误

```bash
# 检查设备权限
ls -la /dev/mali* /dev/dri/

# 添加用户到相应组
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# 重新登录后生效
```

## CUDA移植参考

如果您正在将CUDA程序移植到OpenCL，以下是对应关系：

| CUDA | OpenCL |
|------|--------|
| `__global__` | `__kernel` |
| `__device__` | `__device__` |
| `__shared__` | `__local` |
| `threadIdx.x` | `get_local_id(0)` |
| `blockIdx.x` | `get_group_id(0)` |
| `blockDim.x` | `get_local_size(0)` |
| `gridDim.x` | `get_num_groups(0)` |
| `cudaMalloc` | `clCreateBuffer` |
| `cudaFree` | `clReleaseMemObject` |
| `cudaMemcpy` | `clEnqueueRead/WriteBuffer` |
| `cudaKernelLaunch` | `clEnqueueNDRangeKernel` |
| `__syncthreads()` | `barrier(CLK_LOCAL_MEM_FENCE)` |

### 示例对比

**CUDA Kernel:**
```cuda
__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

**OpenCL Kernel:**
```c
__kernel void add(__global float *a, __global float *b, __global float *c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] + b[i];
}
```

## RK3576上的其他加速方案

除了OpenCL，RK3576还提供以下加速选项：

| 方案 | 用途 | 性能 |
|------|------|------|
| **NPU (RKNN)** | 深度学习推理 | 6 TOPS |
| **OpenCL** | 通用GPU计算 | 145 GFLOPS |
| **RGA** | 2D图像处理 | 硬件加速 |

对于深度学习任务，推荐使用NPU（RKNN SDK）。对于图像处理，RGA可能更高效。

## 许可证

MIT License

## 参考

- [RK3576 Technical Reference Manual](https://www.rock-chips.com/)
- [ARM Mali GPU OpenCL Guide](https://developer.arm.com/solutions/graphics-and-gaming/mali-gpu/best-practices/opencl)
- [OpenCL 2.0 Specification](https://www.khronos.org/opencl/)

# 测试结果记录
(rknn-env) root@myd-lr3576x-debian:/userdata/Projects/yzy/gpu-based-image-stitching-dataset-new/gpu-based-image-stitching/cl_test# ./run_test.sh bench

╔════════════════════════════════════════════════════════╗
║     OpenCL Test Script for MYD-LR3576                  ║
║               Mali-G52 MC3 GPU                         ║
╚════════════════════════════════════════════════════════╝

Checking platform...
✓ Detected RK3576 platform
Checking OpenCL dependencies...
✓ Found OpenCL library: /usr/lib/aarch64-linux-gnu/libOpenCL.so
✓ Found OpenCL library: /usr/lib/aarch64-linux-gnu/libmali.so
✓ Found OpenCL library: /usr/lib/aarch64-linux-gnu/libOpenCL.so
✓ Found OpenCL library: /usr/lib/aarch64-linux-gnu/libmali.so
✓ Found OpenCL ICD configuration
  - mali.icd
✓ clinfo utility available
✓ All dependencies satisfied
Compiling OpenCL test program...
gcc -Wall -Wextra -O2 -std=c99 -o cl_test cl_test.c -lOpenCL -lm
✓ Build successful!

╔════════════════════════════════════════════════════════════╗
║     OpenCL Test for MYD-LR3576 Development Board           ║
║                   Mali-G52 MC3 GPU                         ║
╚════════════════════════════════════════════════════════════╝
arm_release_ver: g13p0-01eac0, rk_so_ver: 10

Found 1 OpenCL platform(s)

╔════════════════════════════════════════════════════════╗
║              OpenCL Platform Information               ║
╚════════════════════════════════════════════════════════╝
  Name:       ARM Platform
  Vendor:     ARM
  Version:    OpenCL 3.0 v1.g13p0-01eac0.0fd2effaec483a5f4c440d2ffa25eb7a
  Profile:    FULL_PROFILE
  Extensions:
    cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics
    cl_khr_local_int32_extended_atomics cl_khr_byte_addressable_store cl_khr_3d_image_writes
    cl_khr_int64_base_atomics cl_khr_int64_extended_atomics cl_khr_fp16
    cl_khr_icd                cl_khr_egl_image          cl_khr_image2d_from_buffer
    cl_khr_depth_images       cl_khr_subgroups          cl_khr_subgroup_extended_types
    cl_khr_subgroup_non_uniform_vote cl_khr_subgroup_ballot    cl_khr_subgroup_non_uniform_arithmetic
    cl_khr_subgroup_shuffle   cl_khr_subgroup_shuffle_relative cl_khr_subgroup_clustered_reduce
    cl_khr_il_program         cl_khr_priority_hints     cl_khr_create_command_queue
    cl_khr_spirv_no_integer_wrap_decoration cl_khr_extended_versioning cl_khr_device_uuid
    cl_khr_suggested_local_work_size cl_khr_extended_bit_ops   cl_khr_integer_dot_product
    cl_khr_semaphore          cl_khr_external_semaphore cl_khr_external_semaphore_sync_fd
    cl_khr_command_buffer     cl_arm_core_id            cl_arm_printf
    cl_arm_non_uniform_work_group_size cl_arm_import_memory      cl_arm_import_memory_dma_buf
    cl_arm_import_memory_host cl_arm_integer_dot_product_int8 cl_arm_job_slot_selection
    cl_arm_scheduling_controls cl_arm_controlled_kernel_termination cl_ext_cxx_for_opencl
    cl_ext_image_tiling_control cl_ext_image_requirements_info cl_ext_image_from_buffer

╔════════════════════════════════════════════════════════╗
║              OpenCL Device Information                 ║
╚════════════════════════════════════════════════════════╝
  Device Name:           Mali-G52 r1p0
  Vendor:                ARM
  Device Version:        OpenCL 3.0 v1.g13p0-01eac0.0fd2effaec483a5f4c440d2ffa25eb7a
  Driver Version:        3.0
  OpenCL C Version:      OpenCL C 3.0 v1.g13p0-01eac0.0fd2effaec483a5f4c440d2ffa25eb7a
  Device Type:           GPU
  Compute Units:         3
  Max Clock Frequency:   900 MHz
  Max Work Group Size:   384
  Max Work Item Dims:    3
  Max Work Item Sizes:   (384, 384, 384)

  --- Memory Information ---
  Global Memory:         7907.29 MB
  Max Alloc Size:        7907.29 MB
  Local Memory:          32.00 KB
  Local Mem Type:        Global
  Max Constant Buffer:   8097068.00 KB
  Global Mem Cache:      256.00 KB
  Cache Line Size:       64 bytes

  --- Image Support ---
  Image Support:         Yes
  Max Image2D Width:     65536
  Max Image2D Height:    65536
  Max Image3D Width:     65536

  --- Floating Point Support ---
  Single Precision FP:   FMA RoundNearest RoundZero RoundInf InfNaN Denorm
  Double Precision FP:   Not Supported

  --- Key Extensions ---
  cl_khr_fp16                         ✓
  cl_khr_fp64                         ✗
  cl_khr_int64_base_atomics           ✓
  cl_khr_int64_extended_atomics       ✓
  cl_khr_local_int32_base_atomics     ✓
  cl_khr_global_int32_base_atomics    ✓
  cl_khr_3d_image_writes              ✓
  cl_khr_image2d_from_buffer          ✓
  cl_khr_subgroups                    ✓
  cl_khr_il_program                   ✓
  cl_arm_core_id                      ✓
  cl_arm_thread_limit_hint            ✗
  cl_arm_non_uniform_work_group_size  ✓

╔════════════════════════════════════════════════════════╗
║           Vector Operations Benchmark                  ║
╚════════════════════════════════════════════════════════╝
  Vector Size: 1048576 elements (4.00 MB)

  Vector Addition:     3.724 ms, 3.38 GB/s
  Vector Multiply:     3.498 ms, 3.60 GB/s
  SAXPY:               3.434 ms, 2.44 GB/s

╔════════════════════════════════════════════════════════╗
║          Matrix Multiplication Benchmark               ║
╚════════════════════════════════════════════════════════╝
  Matrix Size: 512x512 x 512x512
  Operations:  0.27 GFLOP

  Naive MatMul:        345.859 ms, 0.78 GFLOPS
  Tiled MatMul:        66.530 ms, 4.03 GFLOPS

╔════════════════════════════════════════════════════════╗
║            Memory Bandwidth Benchmark                  ║
╚════════════════════════════════════════════════════════╝
  Buffer Size: 128.00 MB

  Host -> Device:      3.02 GB/s
  Device -> Host:      4.12 GB/s
  Device -> Device:    2.46 GB/s

╔════════════════════════════════════════════════════════╗
║              OpenCL Test Completed!                    ║
╚════════════════════════════════════════════════════════╝

