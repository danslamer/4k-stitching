//
// Created by s1nh.org.
//

#include "app.h"

#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

#include "opencv2/core/ocl.hpp"
#include "image_stitcher.h"
#include "stitching_param_generater.h"

// 获取CPU占用率
float getCpuUsage() {
    std::ifstream file("/proc/stat");
    std::string line;
    std::getline(file, line);
    file.close();
    
    std::istringstream iss(line);
    std::string cpu;
    long user, nice, system, idle, iowait, irq, softirq;
    iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq;
    
    long total = user + nice + system + idle + iowait + irq + softirq;
    static long prev_total = 0;
    static long prev_idle = 0;
    
    float usage = 0.0f;
    if (prev_total > 0 && prev_idle > 0) {
        long diff_total = total - prev_total;
        long diff_idle = idle - prev_idle;
        if (diff_total > 0) {
            usage = (1.0f - (float)diff_idle / diff_total) * 100.0f;
        }
    }
    
    prev_total = total;
    prev_idle = idle;
    return usage;
}

// 获取GPU占用率（针对RK3576 Mali GPU）
float getGpuUsage() {
    // 使用RK3576开发板上的正确GPU监控路径
    std::vector<std::string> gpu_files = {
        // RK3576 GPU频率监控
        "/sys/kernel/debug/clk/clk_gpu/clk_rate",
        "/sys/class/devfreq/27800000.gpu/cur_freq",
        // 通用Mali GPU路径
        "/sys/kernel/debug/mali0/utilization",
        "/sys/devices/platform/soc/soc:gpu/mali0/utilization",
        "/sys/kernel/debug/mali/utilization",
        "/sys/devices/platform/mali0/utilization"
    };
    
    for (const auto& path : gpu_files) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::string line;
            if (std::getline(file, line)) {
                try {
                    // 检查是否为频率文件
                    if (path.find("clk_rate") != std::string::npos || path.find("cur_freq") != std::string::npos) {
                        // 频率文件：直接读取频率值
                        float current_freq = std::stof(line);
                        // RK3576 Mali GPU最大频率约为800MHz
                        float usage = (current_freq / 800000000.0f) * 100.0f;
                        file.close();
                        return std::min(usage, 100.0f); // 确保不超过100%
                    } else {
                        // 传统利用率文件
                        float usage = 0.0f;
                        std::istringstream iss(line);
                        std::string token;
                        while (iss >> token) {
                            if (token == "busy:" || token == "utilization:") {
                                if (iss >> token) {
                                    if (token.back() == '%') {
                                        token.pop_back();
                                        usage = std::stof(token);
                                        file.close();
                                        return usage;
                                    }
                                }
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    // 转换失败，继续尝试下一个文件
                }
            }
            file.close();
        }
    }
    
    // 如果上述方法都失败，尝试基于GPU内存使用情况估算
    std::ifstream mem_file("/sys/kernel/debug/mali0/gpu_memory");
    if (mem_file.is_open()) {
        std::string line;
        // 跳过表头
        std::getline(mem_file, line); // <dev> <pages>
        std::getline(mem_file, line); // mali0 <pages_count>
        
        if (line.find("mali0") != std::string::npos) {
            // 提取页面数量
            size_t pos = line.find_last_of(" ");
            if (pos != std::string::npos) {
                std::string pages_str = line.substr(pos + 1);
                try {
                    int pages = std::stoi(pages_str);
                    mem_file.close();
                    
                    // 基于内存使用情况估算占用率
                    // 假设最大内存使用为500000页（约2GB）
                    float usage = (static_cast<float>(pages) / 500000.0f) * 100.0f;
                    return std::min(usage, 100.0f);
                } catch (const std::exception& e) {
                    // 转换失败
                }
            }
        }
        mem_file.close();
    }
    
    return -1.0f; // 无法获取GPU占用率
}


// 获取NPU占用率（针对RK3576 NPU）
float getNpuUsage() {
    // 尝试不同的NPU状态文件路径
    std::vector<std::string> npu_files = {
        "/sys/kernel/debug/rknpu/usage",
        "/sys/devices/platform/soc/soc:npu/usage",
        "/sys/devices/platform/rknpu/usage"
    };
    
    for (const auto& path : npu_files) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::string line;
            float usage = 0.0f;
            while (std::getline(file, line)) {
                if (line.find("usage") != std::string::npos || line.find("utilization") != std::string::npos) {
                    std::istringstream iss(line);
                    std::string key, value;
                    iss >> key >> value;
                    if (value.back() == '%') {
                        value.pop_back();
                        usage = std::stof(value);
                        break;
                    }
                }
            }
            file.close();
            return usage;
        }
    }
    return -1.0f; // 无法获取NPU占用率
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnreachableCode"
using namespace std;

/**
 * 构造函数
 * 初始化传感器数据接口、图像拼接器和相关参数
 */
App::App()
{
  // 初始化视频捕获
  sensorDataInterface_.InitVideoCapture(num_img_);

  // 准备存储第一张图像的向量
  vector<cv::UMat> first_image_vector = vector<cv::UMat>(num_img_);
  vector<cv::Mat> first_mat_vector = vector<cv::Mat>(num_img_);
  
  // 准备存储映射表和ROI区域的向量
  vector<cv::UMat> reproj_xmap_vector;
  vector<cv::UMat> reproj_ymap_vector;
  vector<cv::UMat> undist_xmap_vector;
  vector<cv::UMat> undist_ymap_vector;
  vector<cv::Rect> image_roi_vect;

  // 创建图像互斥锁向量
  vector<mutex> image_mutex_vector(num_img_);
  // 获取第一张图像向量
  sensorDataInterface_.get_image_vector(first_image_vector, image_mutex_vector);

  // 将UMat转换为Mat
  for (size_t i = 0; i < num_img_; ++i)
  {
    first_image_vector[i].copyTo(first_mat_vector[i]);
  }

  // 创建拼接参数生成器并计算参数
  StitchingParamGenerator stitching_param_generator(first_mat_vector);
  stitching_param_generator.GetReprojParams(undist_xmap_vector,
                                            undist_ymap_vector,
                                            reproj_xmap_vector,
                                            reproj_ymap_vector,
                                            image_roi_vect);

  // 设置图像拼接器参数
  image_stitcher_.SetParams(100,  // 融合宽度
                            undist_xmap_vector,
                            undist_ymap_vector,
                            reproj_xmap_vector,
                            reproj_ymap_vector,
                            image_roi_vect);
  
  // 计算总列数
  total_cols_ = 0;
  for (size_t i = 0; i < num_img_; ++i)
  {
    total_cols_ += image_roi_vect[i].width;
  }
  
  // 创建拼接图像
  image_concat_umat_ = cv::UMat(image_roi_vect[0].height, total_cols_, CV_8UC3);
}

/**
 * 运行图像拼接
 * 无限循环执行图像采集、拼接和输出
 */
[[noreturn]] void App::run_stitching()
{
  // 检查并启用OpenCL支持
  bool opencl_supported = false;
  std::cout << "[App] Checking OpenCL support..." << std::endl;
  
  if (cv::ocl::haveOpenCL()) {
    std::cout << "[App] OpenCL is available." << std::endl;
    
    // 尝试创建GPU上下文并获取设备信息
    cv::ocl::Context context;
    if (context.create(cv::ocl::Device::TYPE_GPU)) {
      int device_count = context.ndevices();
      std::cout << "[App] Found " << device_count << " OpenCL device(s):" << std::endl;
      
      for (int i = 0; i < device_count; i++) {
        cv::ocl::Device device = context.device(i);
        std::cout << "[App]   Device " << i << ": " << device.name() << std::endl;
        std::cout << "[App]   Device type: " << device.type() << std::endl;
        std::cout << "[App]   Device is available: " << device.available() << std::endl;
      }
      
      std::cout << "[App] Successfully created OpenCL context!" << std::endl;
      
      // 在上下文创建成功后启用OpenCL
      cv::ocl::setUseOpenCL(true);
      std::cout << "[App] After setUseOpenCL(true): " << cv::ocl::useOpenCL() << std::endl;
      
      if (cv::ocl::useOpenCL()) {
        std::cout << "[App] OpenCL is successfully enabled!" << std::endl;
        opencl_supported = true;
      } else {
        std::cout << "[App] WARNING: OpenCL context created but failed to enable." << std::endl;
        opencl_supported = false;
      }
    } else {
      std::cout << "[App] Failed to create OpenCL context." << std::endl;
    }
  } else {
    std::cout << "[App] OpenCL is not available, falling back to CPU processing." << std::endl;
  }
  
  // 准备存储图像的向量
  vector<cv::UMat> image_vector(num_img_);
  vector<mutex> image_mutex_vector(num_img_);
  vector<cv::UMat> images_warped_vector(num_img_);
  
  // 创建记录视频的线程
  thread record_videos_thread(
      &SensorDataInterface::RecordVideos,
      &sensorDataInterface_);
  
  // 计时变量
  double t0, t1, t2, tn;

  size_t frame_idx = 0;
  // 无限循环
  while (true)
  {
    t0 = cv::getTickCount();

    // 创建变形线程向量
    vector<thread> warp_thread_vect;
    // 获取图像向量
    sensorDataInterface_.get_image_vector(image_vector, image_mutex_vector);
    t1 = cv::getTickCount();

    // 为每个图像创建变形线程
    for (size_t img_idx = 0; img_idx < num_img_; ++img_idx)
    {
      warp_thread_vect.emplace_back(
          &ImageStitcher::WarpImages,
          &image_stitcher_,
          img_idx,
          20,  // 融合像素数
          image_vector,
          ref(image_mutex_vector),
          ref(images_warped_vector),
          ref(image_concat_umat_));
    }
    
    // 等待所有变形线程完成
    for (auto &warp_thread : warp_thread_vect)
    {
      warp_thread.join();
    }
    t2 = cv::getTickCount();

    // 创建掩码以消除渐晕效果
    // 从天空区域（顶部中心区域）取一行像素并拉伸到整个图像作为掩码
    int sky_row = image_concat_umat_.rows * 0.1; // 顶部10%作为天空区域
    cv::UMat sky_row_pixels = image_concat_umat_.row(sky_row).clone();

    // 将该行转换为灰度以创建单通道掩码
    cv::UMat gray_sky_row;
    cv::cvtColor(sky_row_pixels, gray_sky_row, cv::COLOR_BGR2GRAY);

    // 通过在图像的所有行中复制天空行来创建掩码
    cv::UMat mask = cv::UMat::zeros(image_concat_umat_.size(), CV_8UC1);
    for (int y = 0; y < image_concat_umat_.rows; y++)
    {
      gray_sky_row.copyTo(mask.row(y));
    }

    // 归一化掩码，使其最大值为255：mask_refine = 255 - max(mask) + mask
    double min_val, max_val;
    cv::minMaxLoc(mask, &min_val, &max_val);
    cv::UMat mask_refine;
    mask.convertTo(mask_refine, -1, 1.0, 255 - max_val);

    // 确保值保持在[0, 255]范围内
    cv::UMat mask_final;
    cv::max(mask_refine, 0, mask_final);
    cv::min(mask_final, 255, mask_final);

    // 应用掩码：final_stitching = origin_stitching + (255 - mask_refine)
    cv::UMat mask_3channel;
    cv::cvtColor(mask_final, mask_3channel, cv::COLOR_GRAY2BGR);

    // 计算补码：(255 - mask_refine)
    cv::UMat mask_complement;
    cv::subtract(cv::Scalar(255, 255, 255), mask_3channel, mask_complement);

    // 应用校正：final_stitching = origin_stitching + (255 - mask_refine)
    cv::UMat final_output;
    cv::add(image_concat_umat_, mask_complement, final_output);

    // 确保最终值保持在[0, 255]范围内
    cv::UMat final_clamped;
    cv::max(final_output, 0, final_clamped);
    cv::min(final_clamped, 255, final_clamped);

    // 保存拼接结果
    cv::imwrite("../results/image_concat_umat_" + to_string(frame_idx) + ".png",
            final_clamped);

    frame_idx++;
    tn = cv::getTickCount();

    // 获取硬件占用率
    float cpu_usage = getCpuUsage();
    float gpu_usage = getGpuUsage();
    float npu_usage = getNpuUsage();
    
    // 输出性能信息
    cout << "[app] "
         << (t1 - t0) / cv::getTickFrequency() << ";"
         << (t2 - t1) / cv::getTickFrequency() << endl;
    cout << 1 / ((t2 - t0) / cv::getTickFrequency()) << " FPS; "
         << 1 / ((tn - t0) / cv::getTickFrequency()) << " Real FPS." << endl;
    cout << "[app] OpenCL Status: " << (opencl_supported ? "Enabled" : "Disabled") << endl;
    cout << "[app] Hardware Usage: CPU=" << cpu_usage << "%, GPU=" << gpu_usage << "%, NPU=" << npu_usage << "%" << endl;
  }
  
  // 等待记录视频线程完成（实际上不会执行到这里，因为是无限循环）
  record_videos_thread.join();
}

/**
 * 主函数
 * 创建App实例并运行图像拼接
 */
int main()
{
  App app;
  app.run_stitching();
}

#pragma clang diagnostic pop