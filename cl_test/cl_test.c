/**
 * OpenCL Test Program for MYD-LR3576 Development Board
 * 
 * This program tests OpenCL functionality on RK3576 platform
 * with Mali-G52 MC3 GPU (OpenCL 2.0 support).
 * 
 * Features:
 * - Platform and device information detection
 * - OpenCL version and extensions check
 * - Compute performance benchmarks (vector operations, matrix multiply)
 * - Memory bandwidth test
 * - GPU memory info
 * 
 * Compile: make
 * Run: ./cl_test
 */

#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include <unistd.h>
#include <getopt.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

// Test configuration
#define VECTOR_SIZE      (1024 * 1024)   // 1M elements
#define MATRIX_SIZE      512             // 512x512 matrix
#define ITERATIONS       10              // Benchmark iterations
#define MAX_PLATFORMS    8
#define MAX_DEVICES      8
#define MAX_INFO_SIZE    4096

// Color codes for terminal output
#define COLOR_RED     "\033[0;31m"
#define COLOR_GREEN   "\033[0;32m"
#define COLOR_YELLOW  "\033[1;33m"
#define COLOR_BLUE    "\033[0;34m"
#define COLOR_CYAN    "\033[0;36m"
#define COLOR_RESET   "\033[0m"

// ============================================
// OpenCL Kernel Source Code
// ============================================

// Vector addition kernel
static const char *vector_add_kernel =
"__kernel void vector_add(\n"
"    __global const float *a,\n"
"    __global const float *b,\n"
"    __global float *c,\n"
"    const unsigned int n)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < n) {\n"
"        c[id] = a[id] + b[id];\n"
"    }\n"
"}\n";

// Vector multiplication kernel
static const char *vector_mul_kernel =
"__kernel void vector_mul(\n"
"    __global const float *a,\n"
"    __global const float *b,\n"
"    __global float *c,\n"
"    const unsigned int n)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < n) {\n"
"        c[id] = a[id] * b[id];\n"
"    }\n"
"}\n";

// SAXPY kernel (Y = a*X + Y)
static const char *saxpy_kernel =
"__kernel void saxpy(\n"
"    const float a,\n"
"    __global const float *x,\n"
"    __global float *y,\n"
"    const unsigned int n)\n"
"{\n"
"    int id = get_global_id(0);\n"
"    if (id < n) {\n"
"        y[id] = a * x[id] + y[id];\n"
"    }\n"
"}\n";

// Matrix multiplication kernel (naive)
static const char *matmul_kernel =
"__kernel void matmul(\n"
"    __global const float *A,\n"
"    __global const float *B,\n"
"    __global float *C,\n"
"    const unsigned int M,\n"
"    const unsigned int N,\n"
"    const unsigned int K)\n"
"{\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    \n"
"    if (row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < K; k++) {\n"
"            sum += A[row * K + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n";

// Matrix multiplication kernel (tiled, optimized)
static const char *matmul_tiled_kernel =
"__kernel void matmul_tiled(\n"
"    __global const float *A,\n"
"    __global const float *B,\n"
"    __global float *C,\n"
"    const unsigned int M,\n"
"    const unsigned int N,\n"
"    const unsigned int K,\n"
"    __local float *As,\n"
"    __local float *Bs)\n"
"{\n"
"    const int TILE_SIZE = 16;\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    int local_row = get_local_id(0);\n"
"    int local_col = get_local_id(1);\n"
"    \n"
"    float sum = 0.0f;\n"
"    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;\n"
"    \n"
"    for (int t = 0; t < num_tiles; t++) {\n"
"        int a_col = t * TILE_SIZE + local_col;\n"
"        int b_row = t * TILE_SIZE + local_row;\n"
"        \n"
"        As[local_row * TILE_SIZE + local_col] = \n"
"            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;\n"
"        Bs[local_row * TILE_SIZE + local_col] = \n"
"            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;\n"
"        \n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        \n"
"        for (int k = 0; k < TILE_SIZE; k++) {\n"
"            sum += As[local_row * TILE_SIZE + k] * Bs[k * TILE_SIZE + local_col];\n"
"        }\n"
"        \n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    \n"
"    if (row < M && col < N) {\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n";

// ============================================
// Utility Functions
// ============================================

static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static void print_colored(const char *color, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    printf("%s", color);
    vprintf(format, args);
    printf("%s", COLOR_RESET);
    va_end(args);
}

static const char *get_cl_error_string(cl_int error)
{
    static const struct { cl_int code; const char *msg; } errors[] = {
        {CL_SUCCESS, "Success"},
        {CL_DEVICE_NOT_FOUND, "Device not found"},
        {CL_DEVICE_NOT_AVAILABLE, "Device not available"},
        {CL_COMPILER_NOT_AVAILABLE, "Compiler not available"},
        {CL_MEM_OBJECT_ALLOCATION_FAILURE, "Memory allocation failure"},
        {CL_OUT_OF_RESOURCES, "Out of resources"},
        {CL_OUT_OF_HOST_MEMORY, "Out of host memory"},
        {CL_BUILD_PROGRAM_FAILURE, "Build program failure"},
        {CL_INVALID_VALUE, "Invalid value"},
        {CL_INVALID_DEVICE, "Invalid device"},
        {CL_INVALID_CONTEXT, "Invalid context"},
        {CL_INVALID_COMMAND_QUEUE, "Invalid command queue"},
        {CL_INVALID_MEM_OBJECT, "Invalid memory object"},
        {CL_INVALID_PROGRAM, "Invalid program"},
        {CL_INVALID_KERNEL, "Invalid kernel"},
        {CL_INVALID_ARG_INDEX, "Invalid argument index"},
        {CL_INVALID_ARG_VALUE, "Invalid argument value"},
        {CL_INVALID_ARG_SIZE, "Invalid argument size"},
        {CL_INVALID_KERNEL_ARGS, "Invalid kernel arguments"},
        {CL_INVALID_WORK_DIMENSION, "Invalid work dimension"},
        {CL_INVALID_WORK_GROUP_SIZE, "Invalid work group size"},
        {CL_INVALID_WORK_ITEM_SIZE, "Invalid work item size"},
        {CL_INVALID_GLOBAL_OFFSET, "Invalid global offset"},
        {CL_INVALID_BUFFER_SIZE, "Invalid buffer size"},
        {-1, NULL}
    };
    
    for (int i = 0; errors[i].msg; i++) {
        if (errors[i].code == error) {
            return errors[i].msg;
        }
    }
    return "Unknown error";
}

#define CL_CHECK_ERROR(err, msg) \
    do { \
        if ((err) != CL_SUCCESS) { \
            print_colored(COLOR_RED, "Error: %s - %s\n", msg, get_cl_error_string(err)); \
            return err; \
        } \
    } while(0)

// ============================================
// Information Display Functions
// ============================================

static void print_platform_info(cl_platform_id platform)
{
    char info[MAX_INFO_SIZE];
    
    printf("\n");
    print_colored(COLOR_CYAN, "╔════════════════════════════════════════════════════════╗\n");
    print_colored(COLOR_CYAN, "║              OpenCL Platform Information               ║\n");
    print_colored(COLOR_CYAN, "╚════════════════════════════════════════════════════════╝\n");
    
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(info), info, NULL);
    printf("  Name:       %s\n", info);
    
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(info), info, NULL);
    printf("  Vendor:     %s\n", info);
    
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(info), info, NULL);
    printf("  Version:    %s\n", info);
    
    clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, sizeof(info), info, NULL);
    printf("  Profile:    %s\n", info);
    
    clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, sizeof(info), info, NULL);
    printf("  Extensions: \n");
    
    // Print extensions in a formatted way
    char *ext_copy = strdup(info);
    if (ext_copy) {
        char *token = strtok(ext_copy, " ");
        int count = 0;
        while (token) {
            if (count % 3 == 0) printf("    ");
            printf("%-25s ", token);
            if (++count % 3 == 0) printf("\n");
            token = strtok(NULL, " ");
        }
        if (count % 3 != 0) printf("\n");
        free(ext_copy);
    }
}

static void print_device_info(cl_device_id device)
{
    char info[MAX_INFO_SIZE];
    cl_uint uint_val;
    cl_ulong ulong_val;
    size_t size_val;
    cl_bool bool_val;
    cl_device_type type;
    cl_device_local_mem_type local_mem_type;
    cl_device_fp_config fp_config;
    
    printf("\n");
    print_colored(COLOR_CYAN, "╔════════════════════════════════════════════════════════╗\n");
    print_colored(COLOR_CYAN, "║              OpenCL Device Information                 ║\n");
    print_colored(COLOR_CYAN, "╚════════════════════════════════════════════════════════╝\n");
    
    // Basic Info
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info), info, NULL);
    printf("  Device Name:           %s\n", info);
    
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(info), info, NULL);
    printf("  Vendor:                %s\n", info);
    
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(info), info, NULL);
    printf("  Device Version:        %s\n", info);
    
    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(info), info, NULL);
    printf("  Driver Version:        %s\n", info);
    
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(info), info, NULL);
    printf("  OpenCL C Version:      %s\n", info);
    
    // Device Type
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    printf("  Device Type:           ");
    if (type & CL_DEVICE_TYPE_GPU) printf("GPU ");
    if (type & CL_DEVICE_TYPE_CPU) printf("CPU ");
    if (type & CL_DEVICE_TYPE_ACCELERATOR) printf("Accelerator ");
    printf("\n");
    
    // Compute Units & Frequency
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint_val), &uint_val, NULL);
    printf("  Compute Units:         %u\n", uint_val);
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(uint_val), &uint_val, NULL);
    printf("  Max Clock Frequency:   %u MHz\n", uint_val);
    
    // Work Group Info
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_val), &size_val, NULL);
    printf("  Max Work Group Size:   %zu\n", size_val);
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(uint_val), &uint_val, NULL);
    printf("  Max Work Item Dims:    %u\n", uint_val);
    
    size_t work_item_sizes[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(work_item_sizes), work_item_sizes, NULL);
    printf("  Max Work Item Sizes:   (%zu, %zu, %zu)\n", 
           work_item_sizes[0], work_item_sizes[1], work_item_sizes[2]);
    
    // Memory Info
    printf("\n");
    print_colored(COLOR_YELLOW, "  --- Memory Information ---\n");
    
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ulong_val), &ulong_val, NULL);
    printf("  Global Memory:         %.2f MB\n", ulong_val / (1024.0 * 1024.0));
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(ulong_val), &ulong_val, NULL);
    printf("  Max Alloc Size:        %.2f MB\n", ulong_val / (1024.0 * 1024.0));
    
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ulong_val), &ulong_val, NULL);
    printf("  Local Memory:          %.2f KB\n", ulong_val / 1024.0);
    
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
    printf("  Local Mem Type:        %s\n", 
           local_mem_type == CL_LOCAL ? "Local" : "Global");
    
    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(ulong_val), &ulong_val, NULL);
    printf("  Max Constant Buffer:   %.2f KB\n", ulong_val / 1024.0);
    
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(ulong_val), &ulong_val, NULL);
    printf("  Global Mem Cache:      %.2f KB\n", ulong_val / 1024.0);
    
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(uint_val), &uint_val, NULL);
    printf("  Cache Line Size:       %u bytes\n", uint_val);
    
    // Image Support
    printf("\n");
    print_colored(COLOR_YELLOW, "  --- Image Support ---\n");
    
    clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(bool_val), &bool_val, NULL);
    printf("  Image Support:         %s\n", bool_val ? "Yes" : "No");
    
    if (bool_val) {
        clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_val), &size_val, NULL);
        printf("  Max Image2D Width:     %zu\n", size_val);
        
        clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_val), &size_val, NULL);
        printf("  Max Image2D Height:    %zu\n", size_val);
        
        clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_val), &size_val, NULL);
        printf("  Max Image3D Width:     %zu\n", size_val);
    }
    
    // Floating Point Support
    printf("\n");
    print_colored(COLOR_YELLOW, "  --- Floating Point Support ---\n");
    
    clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(fp_config), &fp_config, NULL);
    printf("  Single Precision FP:   ");
    if (fp_config & CL_FP_FMA) printf("FMA ");
    if (fp_config & CL_FP_ROUND_TO_NEAREST) printf("RoundNearest ");
    if (fp_config & CL_FP_ROUND_TO_ZERO) printf("RoundZero ");
    if (fp_config & CL_FP_ROUND_TO_INF) printf("RoundInf ");
    if (fp_config & CL_FP_INF_NAN) printf("InfNaN ");
    if (fp_config & CL_FP_DENORM) printf("Denorm ");
    printf("\n");
    
#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
    clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp_config), &fp_config, NULL);
    printf("  Double Precision FP:   %s\n", fp_config ? "Supported" : "Not Supported");
#endif
    
#ifdef CL_DEVICE_HALF_FP_CONFIG
    clGetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG, sizeof(fp_config), &fp_config, NULL);
    printf("  Half Precision FP:     %s\n", fp_config ? "Supported" : "Not Supported");
#endif
    
    // Extensions
    printf("\n");
    print_colored(COLOR_YELLOW, "  --- Key Extensions ---\n");
    
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(info), info, NULL);
    
    // Check important extensions for RK3576
    struct { const char *name; const char *desc; } key_exts[] = {
        {"cl_khr_fp16", "Half precision FP"},
        {"cl_khr_fp64", "Double precision FP"},
        {"cl_khr_int64_base_atomics", "Int64 base atomics"},
        {"cl_khr_int64_extended_atomics", "Int64 extended atomics"},
        {"cl_khr_local_int32_base_atomics", "Local int32 atomics"},
        {"cl_khr_global_int32_base_atomics", "Global int32 atomics"},
        {"cl_khr_3d_image_writes", "3D image writes"},
        {"cl_khr_image2d_from_buffer", "Image2D from buffer"},
        {"cl_khr_subgroups", "Subgroups"},
        {"cl_khr_il_program", "IL program support"},
        {"cl_arm_core_id", "ARM Core ID"},
        {"cl_arm_thread_limit_hint", "ARM Thread limit"},
        {"cl_arm_non_uniform_work_group_size", "Non-uniform work groups"},
        {NULL, NULL}
    };
    
    for (int i = 0; key_exts[i].name; i++) {
        int supported = strstr(info, key_exts[i].name) != NULL;
        printf("  %-35s %s\n", key_exts[i].name, 
               supported ? COLOR_GREEN "✓" COLOR_RESET : COLOR_RED "✗" COLOR_RESET);
    }
}

// ============================================
// Benchmark Functions
// ============================================

static cl_program create_program(cl_context context, cl_device_id device, const char *source)
{
    cl_int err;
    
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    if (err != CL_SUCCESS) {
        print_colored(COLOR_RED, "Failed to create program: %s\n", get_cl_error_string(err));
        return NULL;
    }
    
    // Build with optimization flags
    const char *build_options = "-cl-fast-relaxed-math -cl-mad-enable";
    err = clBuildProgram(program, 1, &device, build_options, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        print_colored(COLOR_RED, "Failed to build program: %s\n", get_cl_error_string(err));
        
        // Print build log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        
        char *log = malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            printf("Build Log:\n%s\n", log);
            free(log);
        }
        
        clReleaseProgram(program);
        return NULL;
    }
    
    return program;
}

static int benchmark_vector_ops(cl_context context, cl_command_queue queue, cl_device_id device)
{
    (void)device;  // Suppress unused parameter warning
    
    cl_int err;
    cl_program program;
    cl_kernel kernel_add, kernel_mul, kernel_saxpy;
    cl_mem buf_a, buf_b, buf_c;
    float *host_a, *host_b, *host_c;
    size_t size = VECTOR_SIZE;
    size_t bytes = size * sizeof(float);
    
    printf("\n");
    print_colored(COLOR_CYAN, "╔════════════════════════════════════════════════════════╗\n");
    print_colored(COLOR_CYAN, "║           Vector Operations Benchmark                  ║\n");
    print_colored(COLOR_CYAN, "╚════════════════════════════════════════════════════════╝\n");
    
    printf("  Vector Size: %zu elements (%.2f MB)\n", size, bytes / (1024.0 * 1024.0));
    
    // Allocate host memory
    host_a = (float *)malloc(bytes);
    host_b = (float *)malloc(bytes);
    host_c = (float *)malloc(bytes);
    
    if (!host_a || !host_b || !host_c) {
        print_colored(COLOR_RED, "Failed to allocate host memory\n");
        return -1;
    }
    
    // Initialize data
    srand(12345);
    for (size_t i = 0; i < size; i++) {
        host_a[i] = (float)rand() / RAND_MAX;
        host_b[i] = (float)rand() / RAND_MAX;
    }
    
    // Create program with all kernels
    char source[8192];
    snprintf(source, sizeof(source), "%s\n%s\n%s", 
             vector_add_kernel, vector_mul_kernel, saxpy_kernel);
    
    program = create_program(context, device, source);
    if (!program) {
        free(host_a); free(host_b); free(host_c);
        return -1;
    }
    
    // Create kernels
    kernel_add = clCreateKernel(program, "vector_add", &err);
    kernel_mul = clCreateKernel(program, "vector_mul", &err);
    kernel_saxpy = clCreateKernel(program, "saxpy", &err);
    
    // Create buffers
    buf_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, host_a, &err);
    buf_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, host_b, &err);
    buf_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    
    // Benchmark Vector Addition
    printf("\n  %-20s", "Vector Addition:");
    clSetKernelArg(kernel_add, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(kernel_add, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(kernel_add, 2, sizeof(cl_mem), &buf_c);
    clSetKernelArg(kernel_add, 3, sizeof(cl_uint), &size);
    
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        clEnqueueNDRangeKernel(queue, kernel_add, 1, NULL, &size, NULL, 0, NULL, NULL);
    }
    clFinish(queue);
    double end = get_time_ms();
    
    double time_ms = (end - start) / ITERATIONS;
    double bandwidth = (3.0 * bytes) / (time_ms * 1e-3) / 1e9; // 2 read + 1 write
    printf(" %.3f ms, %.2f GB/s\n", time_ms, bandwidth);
    
    // Benchmark Vector Multiplication
    printf("  %-20s", "Vector Multiply:");
    clSetKernelArg(kernel_mul, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), &buf_c);
    clSetKernelArg(kernel_mul, 3, sizeof(cl_uint), &size);
    
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        clEnqueueNDRangeKernel(queue, kernel_mul, 1, NULL, &size, NULL, 0, NULL, NULL);
    }
    clFinish(queue);
    end = get_time_ms();
    
    time_ms = (end - start) / ITERATIONS;
    bandwidth = (3.0 * bytes) / (time_ms * 1e-3) / 1e9;
    printf(" %.3f ms, %.2f GB/s\n", time_ms, bandwidth);
    
    // Benchmark SAXPY
    printf("  %-20s", "SAXPY:");
    float alpha = 2.5f;
    clSetKernelArg(kernel_saxpy, 0, sizeof(float), &alpha);
    clSetKernelArg(kernel_saxpy, 1, sizeof(cl_mem), &buf_a);
    clSetKernelArg(kernel_saxpy, 2, sizeof(cl_mem), &buf_b);
    clSetKernelArg(kernel_saxpy, 3, sizeof(cl_uint), &size);
    
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        clEnqueueNDRangeKernel(queue, kernel_saxpy, 1, NULL, &size, NULL, 0, NULL, NULL);
    }
    clFinish(queue);
    end = get_time_ms();
    
    time_ms = (end - start) / ITERATIONS;
    bandwidth = (2.0 * bytes) / (time_ms * 1e-3) / 1e9; // 1 read + 1 read/write
    printf(" %.3f ms, %.2f GB/s\n", time_ms, bandwidth);
    
    // Cleanup
    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_c);
    clReleaseKernel(kernel_add);
    clReleaseKernel(kernel_mul);
    clReleaseKernel(kernel_saxpy);
    clReleaseProgram(program);
    free(host_a);
    free(host_b);
    free(host_c);
    
    return 0;
}

static int benchmark_matrix_multiply(cl_context context, cl_command_queue queue, cl_device_id device)
{
    cl_int err;
    cl_program program;
    cl_kernel kernel_naive, kernel_tiled;
    cl_mem buf_A, buf_B, buf_C;
    float *host_A, *host_B, *host_C;
    size_t M = MATRIX_SIZE, N = MATRIX_SIZE, K = MATRIX_SIZE;
    size_t bytes = M * N * sizeof(float);
    
    printf("\n");
    print_colored(COLOR_CYAN, "╔════════════════════════════════════════════════════════╗\n");
    print_colored(COLOR_CYAN, "║          Matrix Multiplication Benchmark               ║\n");
    print_colored(COLOR_CYAN, "╚════════════════════════════════════════════════════════╝\n");
    
    printf("  Matrix Size: %zux%zu x %zux%zu\n", M, K, K, N);
    printf("  Operations:  %.2f GFLOP\n", 2.0 * M * N * K / 1e9);
    
    // Allocate host memory
    host_A = (float *)malloc(bytes);
    host_B = (float *)malloc(bytes);
    host_C = (float *)malloc(bytes);
    
    if (!host_A || !host_B || !host_C) {
        print_colored(COLOR_RED, "Failed to allocate host memory\n");
        return -1;
    }
    
    // Initialize data
    srand(12345);
    for (size_t i = 0; i < M * K; i++) host_A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < K * N; i++) host_B[i] = (float)rand() / RAND_MAX;
    
    // Create program
    program = create_program(context, device, matmul_kernel);
    if (!program) {
        free(host_A); free(host_B); free(host_C);
        return -1;
    }
    
    kernel_naive = clCreateKernel(program, "matmul", &err);
    
    // Create buffers
    buf_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, host_A, &err);
    buf_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, host_B, &err);
    buf_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    
    // Set kernel arguments
    clSetKernelArg(kernel_naive, 0, sizeof(cl_mem), &buf_A);
    clSetKernelArg(kernel_naive, 1, sizeof(cl_mem), &buf_B);
    clSetKernelArg(kernel_naive, 2, sizeof(cl_mem), &buf_C);
    clSetKernelArg(kernel_naive, 3, sizeof(cl_uint), &M);
    clSetKernelArg(kernel_naive, 4, sizeof(cl_uint), &N);
    clSetKernelArg(kernel_naive, 5, sizeof(cl_uint), &K);
    
    // Benchmark
    printf("\n  %-20s", "Naive MatMul:");
    
    size_t global_size[2] = {M, N};
    double gflop = 2.0 * M * N * K / 1e9;
    
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        clEnqueueNDRangeKernel(queue, kernel_naive, 2, NULL, global_size, NULL, 0, NULL, NULL);
    }
    clFinish(queue);
    double end = get_time_ms();
    
    double time_ms = (end - start) / ITERATIONS;
    double gflops = gflop / (time_ms * 1e-3);
    printf(" %.3f ms, %.2f GFLOPS\n", time_ms, gflops);
    
    // Try tiled version
    clReleaseKernel(kernel_naive);
    clReleaseProgram(program);
    
    program = create_program(context, device, matmul_tiled_kernel);
    if (program) {
        kernel_tiled = clCreateKernel(program, "matmul_tiled", &err);
        
        size_t local_size[2] = {16, 16};
        size_t tiled_global[2] = {((M + 15) / 16) * 16, ((N + 15) / 16) * 16};
        
        clSetKernelArg(kernel_tiled, 0, sizeof(cl_mem), &buf_A);
        clSetKernelArg(kernel_tiled, 1, sizeof(cl_mem), &buf_B);
        clSetKernelArg(kernel_tiled, 2, sizeof(cl_mem), &buf_C);
        clSetKernelArg(kernel_tiled, 3, sizeof(cl_uint), &M);
        clSetKernelArg(kernel_tiled, 4, sizeof(cl_uint), &N);
        clSetKernelArg(kernel_tiled, 5, sizeof(cl_uint), &K);
        clSetKernelArg(kernel_tiled, 6, 16 * 16 * sizeof(float), NULL);  // Local memory A
        clSetKernelArg(kernel_tiled, 7, 16 * 16 * sizeof(float), NULL);  // Local memory B
        
        printf("  %-20s", "Tiled MatMul:");
        
        start = get_time_ms();
        for (int i = 0; i < ITERATIONS; i++) {
            clEnqueueNDRangeKernel(queue, kernel_tiled, 2, NULL, tiled_global, local_size, 0, NULL, NULL);
        }
        clFinish(queue);
        end = get_time_ms();
        
        time_ms = (end - start) / ITERATIONS;
        gflops = gflop / (time_ms * 1e-3);
        printf(" %.3f ms, %.2f GFLOPS\n", time_ms, gflops);
        
        clReleaseKernel(kernel_tiled);
        clReleaseProgram(program);
    }
    
    // Cleanup
    clReleaseMemObject(buf_A);
    clReleaseMemObject(buf_B);
    clReleaseMemObject(buf_C);
    free(host_A);
    free(host_B);
    free(host_C);
    
    return 0;
}

static int benchmark_memory_bandwidth(cl_context context, cl_command_queue queue)
{
    cl_int err;
    cl_mem buf_src, buf_dst;
    float *host_data;
    size_t size = 32 * 1024 * 1024;  // 32M elements = 128MB
    size_t bytes = size * sizeof(float);
    
    printf("\n");
    print_colored(COLOR_CYAN, "╔════════════════════════════════════════════════════════╗\n");
    print_colored(COLOR_CYAN, "║            Memory Bandwidth Benchmark                  ║\n");
    print_colored(COLOR_CYAN, "╚════════════════════════════════════════════════════════╝\n");
    
    printf("  Buffer Size: %.2f MB\n", bytes / (1024.0 * 1024.0));
    
    // Allocate host memory
    host_data = (float *)malloc(bytes);
    if (!host_data) {
        print_colored(COLOR_RED, "Failed to allocate host memory\n");
        return -1;
    }
    
    for (size_t i = 0; i < size; i++) {
        host_data[i] = (float)i;
    }
    
    // Create buffers
    buf_src = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, host_data, &err);
    buf_dst = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    
    if (err != CL_SUCCESS) {
        print_colored(COLOR_RED, "Failed to create buffers: %s\n", get_cl_error_string(err));
        free(host_data);
        return -1;
    }
    
    // Test Host to Device bandwidth
    printf("\n  %-20s", "Host -> Device:");
    double start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        clEnqueueWriteBuffer(queue, buf_src, CL_TRUE, 0, bytes, host_data, 0, NULL, NULL);
    }
    double end = get_time_ms();
    double bandwidth = (double)bytes / ((end - start) / ITERATIONS / 1000.0) / 1e9;
    printf(" %.2f GB/s\n", bandwidth);
    
    // Test Device to Host bandwidth
    printf("  %-20s", "Device -> Host:");
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        clEnqueueReadBuffer(queue, buf_src, CL_TRUE, 0, bytes, host_data, 0, NULL, NULL);
    }
    end = get_time_ms();
    bandwidth = (double)bytes / ((end - start) / ITERATIONS / 1000.0) / 1e9;
    printf(" %.2f GB/s\n", bandwidth);
    
    // Test Device to Device (copy)
    printf("  %-20s", "Device -> Device:");
    start = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        clEnqueueCopyBuffer(queue, buf_src, buf_dst, 0, 0, bytes, 0, NULL, NULL);
    }
    clFinish(queue);
    end = get_time_ms();
    bandwidth = (double)bytes / ((end - start) / ITERATIONS / 1000.0) / 1e9;
    printf(" %.2f GB/s\n", bandwidth);
    
    // Cleanup
    clReleaseMemObject(buf_src);
    clReleaseMemObject(buf_dst);
    free(host_data);
    
    return 0;
}

// ============================================
// Main Function
// ============================================

static void print_usage(const char *program)
{
    printf("Usage: %s [options]\n\n", program);
    printf("OpenCL Test Program for RK3576 Development Board\n\n");
    printf("Options:\n");
    printf("  -h, --help        Show this help message\n");
    printf("  -i, --info        Show platform/device info only\n");
    printf("  -b, --benchmark   Run performance benchmarks\n");
    printf("  -v, --vector      Run vector operations benchmark\n");
    printf("  -m, --matrix      Run matrix multiplication benchmark\n");
    printf("  -M, --memory      Run memory bandwidth benchmark\n");
    printf("  -a, --all         Run all tests (default)\n");
    printf("\nExamples:\n");
    printf("  %s -i             # Show GPU info only\n", program);
    printf("  %s -b             # Run all benchmarks\n", program);
    printf("  %s -v -m          # Run vector and matrix tests\n", program);
}

int main(int argc, char *argv[])
{
    cl_int err;
    cl_platform_id platforms[MAX_PLATFORMS];
    cl_device_id devices[MAX_DEVICES];
    cl_uint num_platforms, num_devices;
    cl_context context;
    cl_command_queue queue;
    
    int show_info = 0;
    int run_benchmark = 0;
    int run_vector = 0;
    int run_matrix = 0;
    int run_memory = 0;
    
    // Parse arguments
    static struct option long_options[] = {
        {"help",       no_argument, 0, 'h'},
        {"info",       no_argument, 0, 'i'},
        {"benchmark",  no_argument, 0, 'b'},
        {"vector",     no_argument, 0, 'v'},
        {"matrix",     no_argument, 0, 'm'},
        {"memory",     no_argument, 0, 'M'},
        {"all",        no_argument, 0, 'a'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "hibvmMa", long_options, NULL)) != -1) {
        switch (opt) {
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'i':
                show_info = 1;
                break;
            case 'b':
                run_benchmark = 1;
                break;
            case 'v':
                run_vector = 1;
                break;
            case 'm':
                run_matrix = 1;
                break;
            case 'M':
                run_memory = 1;
                break;
            case 'a':
                show_info = 1;
                run_benchmark = 1;
                run_vector = 1;
                run_matrix = 1;
                run_memory = 1;
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Default: show info if no options specified
    if (!show_info && !run_benchmark && !run_vector && !run_matrix && !run_memory) {
        show_info = 1;
        run_benchmark = 1;
        run_vector = 1;
        run_matrix = 1;
        run_memory = 1;
    }
    
    // Print header
    printf("\n");
    print_colored(COLOR_BLUE, "╔════════════════════════════════════════════════════════════╗\n");
    print_colored(COLOR_BLUE, "║     OpenCL Test for MYD-LR3576 Development Board           ║\n");
    print_colored(COLOR_BLUE, "║                   Mali-G52 MC3 GPU                         ║\n");
    print_colored(COLOR_BLUE, "╚════════════════════════════════════════════════════════════╝\n");
    
    // Get platforms
    err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        print_colored(COLOR_RED, "\nNo OpenCL platforms found!\n");
        print_colored(COLOR_YELLOW, "Please ensure OpenCL drivers are installed:\n");
        printf("  - Mali GPU drivers for RK3576\n");
        printf("  - Or install: sudo apt install ocl-icd-opencl-dev\n");
        return 1;
    }
    
    printf("\nFound %d OpenCL platform(s)\n", num_platforms);
    
    // Find GPU device
    cl_device_id gpu_device = NULL;
    cl_platform_id gpu_platform = NULL;
    
    for (cl_uint i = 0; i < num_platforms; i++) {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            gpu_device = devices[0];
            gpu_platform = platforms[i];
            break;
        }
    }
    
    // Fallback to any device if no GPU found
    if (!gpu_device) {
        print_colored(COLOR_YELLOW, "No GPU device found, trying any device...\n");
        for (cl_uint i = 0; i < num_platforms; i++) {
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &num_devices);
            if (err == CL_SUCCESS && num_devices > 0) {
                gpu_device = devices[0];
                gpu_platform = platforms[i];
                break;
            }
        }
    }
    
    if (!gpu_device) {
        print_colored(COLOR_RED, "\nNo OpenCL device found!\n");
        return 1;
    }
    
    // Show info
    if (show_info) {
        print_platform_info(gpu_platform);
        print_device_info(gpu_device);
    }
    
    // Create context and queue for benchmarks
    if (run_benchmark || run_vector || run_matrix || run_memory) {
        context = clCreateContext(NULL, 1, &gpu_device, NULL, NULL, &err);
        if (err != CL_SUCCESS) {
            print_colored(COLOR_RED, "Failed to create OpenCL context: %s\n", get_cl_error_string(err));
            return 1;
        }
        
        // Use deprecated API for compatibility with older OpenCL headers
        queue = clCreateCommandQueue(context, gpu_device, 0, &err);
        if (err != CL_SUCCESS) {
            print_colored(COLOR_RED, "Failed to create command queue: %s\n", get_cl_error_string(err));
            clReleaseContext(context);
            return 1;
        }
        
        // Run benchmarks
        if (run_vector || run_benchmark) {
            benchmark_vector_ops(context, queue, gpu_device);
        }
        
        if (run_matrix || run_benchmark) {
            benchmark_matrix_multiply(context, queue, gpu_device);
        }
        
        if (run_memory || run_benchmark) {
            benchmark_memory_bandwidth(context, queue);
        }
        
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
    
    // Summary
    printf("\n");
    print_colored(COLOR_GREEN, "╔════════════════════════════════════════════════════════╗\n");
    print_colored(COLOR_GREEN, "║              OpenCL Test Completed!                    ║\n");
    print_colored(COLOR_GREEN, "╚════════════════════════════════════════════════════════╝\n\n");
    
    return 0;
}
