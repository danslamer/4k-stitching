//
// Created by s1nh.org on 2020/12/1.
//

#include "image_stitcher.h"

#include <thread>
#include <mutex>

#include "opencv2/core/ocl.hpp"

/**
 * 设置拼接参数
 * @param blend_width 融合宽度
 * @param undist_xmap_vector 去畸变x方向映射表
 * @param undist_ymap_vector 去畸变y方向映射表
 * @param reproj_xmap_vector 重投影x方向映射表
 * @param reproj_ymap_vector 重投影y方向映射表
 * @param projected_image_roi_vect_refined 精细化后的投影图像ROI区域
 */
void ImageStitcher::SetParams(
    const int& blend_width,
    vector<cv::UMat>& undist_xmap_vector,
    vector<cv::UMat>& undist_ymap_vector,
    vector<cv::UMat>& reproj_xmap_vector,
    vector<cv::UMat>& reproj_ymap_vector,
    vector<cv::Rect>& projected_image_roi_vect_refined) {
  // 显式初始化和启用OpenCL
  std::cout << "[ImageStitcher] Initializing OpenCL..." << std::endl;
  if (cv::ocl::haveOpenCL()) {
    // 显式创建OpenCL上下文
    cv::ocl::Context context;
    if (context.create(cv::ocl::Device::TYPE_GPU)) {
      std::cout << "[ImageStitcher] OpenCL context created with " << context.ndevices() << " device(s)." << std::endl;
      for (int i = 0; i < context.ndevices(); i++) {
        cv::ocl::Device device = context.device(i);
        std::cout << "[ImageStitcher]   Device " << i << ": " << device.name() << std::endl;
      }
      
      // 在上下文创建成功后启用OpenCL
      cv::ocl::setUseOpenCL(true);
      std::cout << "[ImageStitcher] After setUseOpenCL(true): " << cv::ocl::useOpenCL() << std::endl;
      
      if (cv::ocl::useOpenCL()) {
        std::cout << "[ImageStitcher] OpenCL is enabled." << std::endl;
      } else {
        std::cout << "[ImageStitcher] WARNING: Failed to enable OpenCL." << std::endl;
      }
    } else {
      std::cout << "[ImageStitcher] Failed to create OpenCL context." << std::endl;
    }
  } else {
    std::cout << "[ImageStitcher] OpenCL is not available." << std::endl;
  }
  // 初始化图像数量
  num_img_ = undist_xmap_vector.size();
  // 初始化互斥锁向量
  warp_mutex_vector_ = vector<mutex>(num_img_);

  // 复制映射表和ROI区域
  undist_xmap_vector_ = undist_xmap_vector;
  undist_ymap_vector_ = undist_ymap_vector;
  reproj_xmap_vector_ = reproj_xmap_vector;
  reproj_ymap_vector_ = reproj_ymap_vector;
  roi_vect_ = projected_image_roi_vect_refined;

  // 合并两个重映射操作（为了稍微加快速度）
  final_xmap_vector_ = vector<cv::UMat>(undist_ymap_vector.size());
  final_ymap_vector_ = vector<cv::UMat>(undist_ymap_vector.size());
  tmp_umat_vect_ = vector<cv::UMat>(undist_ymap_vector.size());
  
  // 为每个图像计算最终的映射表
  for (size_t img_idx = 0; img_idx < num_img_; ++img_idx) {
    // 对x方向映射表应用重投影
    remap(undist_xmap_vector_[img_idx],
          final_xmap_vector_[img_idx],
          reproj_xmap_vector_[img_idx],
          reproj_ymap_vector_[img_idx],
          cv::INTER_LINEAR);
    // 对y方向映射表应用重投影
    remap(undist_ymap_vector_[img_idx],
          final_ymap_vector_[img_idx],
          reproj_xmap_vector_[img_idx],
          reproj_ymap_vector_[img_idx],
          cv::INTER_LINEAR);
    
    // 创建临时UMat
    cv::UMat _;
    undist_xmap_vector[img_idx].copyTo(_);
//    wrap_vec_.push_back(_);//TODO: Use zeros instead of this fake data.
  }
  
  // 创建权重图
  CreateWeightMap(undist_ymap_vector[0].rows, blend_width);
}


/**
 * 创建权重图
 * @param height 高度
 * @param width 宽度
 */
void ImageStitcher::CreateWeightMap(const int& height, const int& width) {
  // TODO: 尝试使用CV_16F类型
  
  // 创建左侧权重图（从左到右，权重从0到255）
  cv::Mat _l = cv::Mat(height, width, CV_8UC3);
  // 创建右侧权重图（从左到右，权重从255到0）
  cv::Mat _r = cv::Mat(height, width, CV_8UC3);
  
  // 填充权重图
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      // 左侧权重图：像素位置越靠右，权重越大
      _l.at<cv::Vec3b>(i, j)[0] =
      _l.at<cv::Vec3b>(i, j)[1] =
      _l.at<cv::Vec3b>(i, j)[2] =
          cv::saturate_cast<uchar>((float) j / (float) width * 255);

      // 右侧权重图：像素位置越靠左，权重越大
      _r.at<cv::Vec3b>(i, j)[0] =
      _r.at<cv::Vec3b>(i, j)[1] =
      _r.at<cv::Vec3b>(i, j)[2] =
          cv::saturate_cast<uchar>((float) (width - j) / (float) width * 255);

    }
  }
  
  // 将权重图添加到权重图向量中
  weightMap_.emplace_back(_l.getUMat(cv::ACCESS_READ));
  weightMap_.emplace_back(_r.getUMat(cv::ACCESS_READ));

  // 保存权重图到文件
  cv::imwrite("../results/_weight_map_l.png", weightMap_[0]);
  cv::imwrite("../results/_weight_map_r.png", weightMap_[1]);
}

/**
 * 对图像进行变形和融合
 * @param img_idx 图像索引
 * @param fusion_pixel 融合像素数
 * @param image_vector 输入图像向量
 * @param image_mutex_vector 图像互斥锁向量
 * @param images_warped_with_roi_vector 带ROI的变形图像向量
 * @param image_concat_umat 拼接后的图像
 */
void ImageStitcher::WarpImages(
    const int& img_idx,
    const int& fusion_pixel,
    const vector<cv::UMat>& image_vector,
    vector<mutex>& image_mutex_vector,
    vector<cv::UMat>& images_warped_with_roi_vector,
    cv::UMat& image_concat_umat) {
  // 计时变量
  double t0, t1, t2, t3, tn;
  t0 = cv::getTickCount();
  
  // 锁定当前图像的互斥锁
  image_mutex_vector[img_idx].lock();

  // 记录开始时间
  t1 = cv::getTickCount();

  // 检查当前是否使用OpenCL
  bool using_opencl = cv::ocl::useOpenCL();
  std::cout << "[ImageStitcher] Warping image " << img_idx << " using " << (using_opencl ? "GPU" : "CPU") << std::endl;
  
  // 检查UMat类型
  std::cout << "[ImageStitcher] Using UMat for image processing" << std::endl;
  
  // 强制使用GPU进行计算
  cv::UMat src = image_vector[img_idx].clone();
  cv::UMat dst;
  
  // 使用最终映射表对图像进行变形（合并了去畸变和重投影）
  remap(src,
        dst,
        final_xmap_vector_[img_idx],
        final_ymap_vector_[img_idx],
        cv::INTER_LINEAR);
  
  // 将结果复制回tmp_umat_vect_[img_idx]
  dst.copyTo(tmp_umat_vect_[img_idx]);
  
  // 检查操作是否在GPU上执行
  std::cout << "[ImageStitcher] Warp operation completed." << std::endl;
  
  // 解锁当前图像的互斥锁
  image_mutex_vector[img_idx].unlock();
  t2 = cv::getTickCount();
  t3 = cv::getTickCount();

  // 融合两张图像的边缘
  if (img_idx > 0) {
    // 检查权重图是否有效
    if (weightMap_.empty() || weightMap_[0].cols <= 0 || weightMap_[0].rows <= 0) {
      std::cout << "[ImageStitcher] Weight map is invalid, skipping blending." << std::endl;
      return;
    }
    
    // 根据重叠区域确定实际融合所需的大小
    int blend_width_actual = min(weightMap_[0].cols, min(tmp_umat_vect_[img_idx].cols, tmp_umat_vect_[img_idx-1].cols));
    int blend_height_actual = min(weightMap_[0].rows, min(tmp_umat_vect_[img_idx].rows, tmp_umat_vect_[img_idx-1].rows));
    
    // 确保融合宽度和高度为正
    if (blend_width_actual <= 0 || blend_height_actual <= 0) {
      std::cout << "[ImageStitcher] Invalid blend size, skipping blending." << std::endl;
      return;
    }
    
    // 创建安全的ROI矩形，避免越界访问
    cv::Rect roi_r(roi_vect_[img_idx].x, roi_vect_[img_idx].y, 
                   blend_width_actual, blend_height_actual);
    // 确保ROI在右侧图像的边界内
    roi_r.x = max(0, min(roi_r.x, tmp_umat_vect_[img_idx].cols - blend_width_actual));
    roi_r.y = max(0, min(roi_r.y, tmp_umat_vect_[img_idx].rows - blend_height_actual));
    roi_r.width = min(roi_r.width, tmp_umat_vect_[img_idx].cols - roi_r.x);
    roi_r.height = min(roi_r.height, tmp_umat_vect_[img_idx].rows - roi_r.y);
    
    // 确保ROI有效
    if (roi_r.width <= 0 || roi_r.height <= 0) {
      std::cout << "[ImageStitcher] Invalid ROI for right image, skipping blending." << std::endl;
      return;
    }
    
    // 提取右侧图像的融合区域
    cv::UMat _r = tmp_umat_vect_[img_idx](roi_r);

    // 锁定左侧图像的互斥锁
    warp_mutex_vector_[img_idx - 1].lock();
    
    // 创建左侧图像的融合区域
    cv::Rect roi_l(roi_vect_[img_idx - 1].x + roi_vect_[img_idx - 1].width,
                   roi_vect_[img_idx - 1].y, blend_width_actual, blend_height_actual);
    // 确保ROI在左侧图像的边界内
    roi_l.x = max(0, min(roi_l.x, tmp_umat_vect_[img_idx - 1].cols - blend_width_actual));
    roi_l.y = max(0, min(roi_l.y, tmp_umat_vect_[img_idx - 1].rows - blend_height_actual));
    roi_l.width = min(roi_l.width, tmp_umat_vect_[img_idx - 1].cols - roi_l.x);
    roi_l.height = min(roi_l.height, tmp_umat_vect_[img_idx - 1].rows - roi_l.y);
    
    // 确保ROI有效
    if (roi_l.width <= 0 || roi_l.height <= 0) {
      std::cout << "[ImageStitcher] Invalid ROI for left image, skipping blending." << std::endl;
      warp_mutex_vector_[img_idx - 1].unlock();
      return;
    }
    
    // 提取左侧图像的融合区域
    cv::UMat _l = tmp_umat_vect_[img_idx - 1](roi_l);
    
    // 如果需要，调整权重图大小以匹配实际ROI大小
    cv::UMat resized_weight_0, resized_weight_1;
    if (weightMap_[0].cols != _r.cols || weightMap_[0].rows != _r.rows) {
      // 确保目标大小有效
      if (_r.cols > 0 && _r.rows > 0) {
        cv::resize(weightMap_[0], resized_weight_0, cv::Size(_r.cols, _r.rows));
      } else {
        std::cout << "[ImageStitcher] Invalid right image size, skipping blending." << std::endl;
        warp_mutex_vector_[img_idx - 1].unlock();
        return;
      }
      if (_l.cols > 0 && _l.rows > 0) {
        cv::resize(weightMap_[1], resized_weight_1, cv::Size(_l.cols, _l.rows));
      } else {
        std::cout << "[ImageStitcher] Invalid left image size, skipping blending." << std::endl;
        warp_mutex_vector_[img_idx - 1].unlock();
        return;
      }
    } else {
      resized_weight_0 = weightMap_[0];
      resized_weight_1 = weightMap_[1];
    }
    
    // 解锁左侧图像的互斥锁
    warp_mutex_vector_[img_idx - 1].unlock();

    // 应用权重图进行融合
    cv::multiply(_r, resized_weight_0, _r, 1. / 255.);
    cv::multiply(_l, resized_weight_1, _l, 1. / 255.);
    cv::add(_r, _l, _r);
  }

  // 应用ROI，带边界检查
  int cols = 0;
  // 计算当前图像在拼接图像中的起始列
  for (size_t i = 0; i < img_idx; i++) {
    cols += roi_vect_[i].width;
  }

  // 确保ROI在源图像的边界内
  cv::Rect safe_roi = roi_vect_[img_idx];
  safe_roi.x = max(0, min(safe_roi.x, tmp_umat_vect_[img_idx].cols - 1));
  safe_roi.y = max(0, min(safe_roi.y, tmp_umat_vect_[img_idx].rows - 1));
  safe_roi.width = max(0, min(safe_roi.width, tmp_umat_vect_[img_idx].cols - safe_roi.x));
  safe_roi.height = max(0, min(safe_roi.height, tmp_umat_vect_[img_idx].rows - safe_roi.y));

  // 确保目标矩形在边界内
  cv::Rect dest_rect = cv::Rect(cols, 0, roi_vect_[img_idx].width, roi_vect_[img_idx].height);
  dest_rect.width = max(0, min(dest_rect.width, image_concat_umat.cols - cols));
  dest_rect.height = max(0, min(dest_rect.height, image_concat_umat.rows));

  // 额外检查：确保源和目标具有相同的大小
  if (safe_roi.width > 0 && safe_roi.height > 0 && 
      dest_rect.width > 0 && dest_rect.height > 0) {
    // 调整dest_rect以匹配源ROI的实际大小
    dest_rect.width = min(dest_rect.width, safe_roi.width);
    dest_rect.height = min(dest_rect.height, safe_roi.height);
    
    // 确保ROI和目标矩形都在各自图像的边界内
    if (safe_roi.x + safe_roi.width <= tmp_umat_vect_[img_idx].cols &&
        safe_roi.y + safe_roi.height <= tmp_umat_vect_[img_idx].rows &&
        dest_rect.x + dest_rect.width <= image_concat_umat.cols &&
        dest_rect.y + dest_rect.height <= image_concat_umat.rows) {
      // 提取源ROI和目标ROI
      cv::UMat src_roi = tmp_umat_vect_[img_idx](safe_roi);
      cv::UMat dst_roi = image_concat_umat(dest_rect);
      
      // 只有当尺寸匹配时才进行复制
      if (src_roi.size() == dst_roi.size()) {
        src_roi.copyTo(dst_roi);
      }
    }
  }

  // 记录结束时间并输出性能信息
  tn = cv::getTickCount();
  cout << "[image_stitcher] "
       << (t1 - t0) / cv::getTickFrequency() << ";"
       << (t2 - t1) / cv::getTickFrequency() << ";"
       << (t3 - t2) / cv::getTickFrequency() << ";"
       << 1 / (tn - t0) * cv::getTickFrequency() << endl;
}