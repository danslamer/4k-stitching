//
// Created by s1nh.org on 2020/11/13.
// Modified from samples/cpp/stitching_detailed.cpp
//

#include "stitching_param_generater.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

#include "opencv2/core/ocl.hpp"


using namespace std;
using namespace cv;
using namespace cv::detail;

namespace {

std::string GetEnvOrDefault(const char* name, const std::string& default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0')
    return default_value;
  return value;
}

}  // namespace

#define ENABLE_LOG 0
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

// 检查并启用OpenCL支持的函数
bool checkOpenCLSupport() {
  std::cout << "[StitchingParamGenerator] Checking OpenCL support..." << std::endl;
  
  if (cv::ocl::haveOpenCL()) {
    std::cout << "[StitchingParamGenerator] OpenCL is available." << std::endl;
    
    // 尝试创建默认上下文并获取设备信息
    cv::ocl::Context context;
    if (context.create(cv::ocl::Device::TYPE_GPU)) {
      int device_count = context.ndevices();
      std::cout << "[StitchingParamGenerator] Found " << device_count << " OpenCL device(s):" << std::endl;
      
      for (int i = 0; i < device_count; i++) {
        cv::ocl::Device device = context.device(i);
        std::cout << "[StitchingParamGenerator]   Device " << i << ": " << device.name() << std::endl;
        std::cout << "[StitchingParamGenerator]   Device type: " << device.type() << std::endl;
        std::cout << "[StitchingParamGenerator]   Device is available: " << device.available() << std::endl;
      }
      
      std::cout << "[StitchingParamGenerator] Successfully created OpenCL context!" << std::endl;
      
      // 在上下文创建成功后启用OpenCL
      cv::ocl::setUseOpenCL(true);
      std::cout << "[StitchingParamGenerator] After setUseOpenCL(true): " << cv::ocl::useOpenCL() << std::endl;
      
      if (cv::ocl::useOpenCL()) {
        std::cout << "[StitchingParamGenerator] OpenCL has been enabled." << std::endl;
        return true;
      } else {
        std::cout << "[StitchingParamGenerator] WARNING: Failed to enable OpenCL, falling back to CPU." << std::endl;
        return false;
      }
    } else {
      std::cout << "[StitchingParamGenerator] Failed to create OpenCL context." << std::endl;
    }
    
  } else {
    std::cout << "[StitchingParamGenerator] OpenCL is not available, falling back to CPU processing." << std::endl;
  }
  return false;
}


/**
 * StitchingParamGenerator构造函数
 * @param image_vector 输入图像向量，包含要拼接的所有图像
 */
StitchingParamGenerator::StitchingParamGenerator(
    const vector<cv::Mat>& image_vector) {
  params_dir_ = GetEnvOrDefault("STITCH_PARAMS_DIR", "../params");
  // 初始化图像数量
  num_img_ = image_vector.size();

  // 检测OpenCL支持
  bool opencl_available = checkOpenCLSupport();
  if (opencl_available) {
    LOGLN("OpenCL is available, enabling GPU acceleration.");
  } else {
    LOGLN("OpenCL is not available, falling back to CPU processing.");
  }

  // 复制输入图像向量
  image_vector_ = image_vector;
  // 初始化各种向量容器
  mask_vector_ = vector<cv::UMat>(num_img_);
  mask_warped_vector_ = vector<cv::UMat>(num_img_);
  image_size_vector_ = vector<cv::Size>(num_img_);
  image_warped_size_vector_ = vector<cv::Size>(num_img_);
  reproj_xmap_vector_ = vector<cv::UMat>(num_img_);
  reproj_ymap_vector_ = vector<cv::UMat>(num_img_);
  camera_params_vector_ = vector<cv::detail::CameraParams>(camera_params_vector_);

  // 初始化精细化后的投影图像ROI区域
  projected_image_roi_vect_refined_ = vector<cv::Rect>(num_img_);

  // 记录每个图像的尺寸
  for (size_t img_idx = 0; img_idx < num_img_; img_idx++) {
    image_size_vector_[img_idx] = image_vector_[img_idx].size();
  }

  // 初始化去畸变映射表
  InitUndistortMap();

  // 对每个图像应用去畸变
  for (size_t img_idx = 0; img_idx < num_img_; ++img_idx) {
    cv::remap(image_vector_[img_idx],  // 输入图像
              image_vector_[img_idx],  // 输出图像
              undist_xmap_vector_[img_idx],  // x方向去畸变映射表
              undist_ymap_vector_[img_idx],  // y方向去畸变映射表
              cv::INTER_LINEAR);  // 线性插值
  }

  // 初始化相机参数
  InitCameraParam();
  // 初始化变形器
  InitWarper();
}

/**
 * 初始化相机参数
 * 包括特征点检测、匹配、相机参数估计和光束法平差
 */
void StitchingParamGenerator::InitCameraParam() {
  // 创建特征检测器，使用SIFT算法
  Ptr<Feature2D> finder;
  finder = SIFT::create();
  
  // 存储每个图像的特征
  vector<ImageFeatures> features(num_img_);
  vector<Size> full_img_sizes(num_img_);
  
  // 为每个图像提取特征点
  for (int i = 0; i < num_img_; ++i) {
    computeImageFeatures(finder, image_vector_[i], features[i]);
    features[i].img_idx = i;
    LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());
  }
  
  LOG("Pairwise matching");
  // 存储图像对之间的匹配信息
  vector<MatchesInfo> pairwise_matches;
  
  // 创建特征匹配器（使用OpenCL加速）
  Ptr<FeaturesMatcher> matcher;
  if (matcher_type == "affine") {
    LOGLN("Using AffineBestOf2NearestMatcher");
    matcher = makePtr<AffineBestOf2NearestMatcher>(false, false, match_conf); // 禁用CUDA
  } else if (range_width == -1) {
    LOGLN("Using BestOf2NearestMatcher");
    matcher = makePtr<BestOf2NearestMatcher>(false, match_conf); // 禁用CUDA
  } else {
    LOGLN("Using BestOf2NearestRangeMatcher");
    matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, false, match_conf); // 禁用CUDA
  }
  
  // 执行特征匹配
  (*matcher)(features, pairwise_matches);
  // 释放匹配器的内存
  matcher->collectGarbage();
  
  // 检查是否需要保存匹配图
  if (save_graph) {
    LOGLN("Saving matches graph...");
    ofstream f(save_graph_to.c_str());
    f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
  }
  
  // 创建相机参数估计器
  Ptr<Estimator> estimator;
  if (estimator_type == "affine")
    estimator = makePtr<AffineBasedEstimator>();
  else
    estimator = makePtr<HomographyBasedEstimator>();
  
  // 估计相机参数
  if (!(*estimator)(features, pairwise_matches, camera_params_vector_)) {
    cout << "Homography estimation failed.\n";
    assert(false);
  }
  
  // 转换相机旋转矩阵为32位浮点数
  for (auto& i : camera_params_vector_) {
    Mat R;
    i.R.convertTo(R, CV_32F);
    i.R = R;
  }
  
  // 创建光束法平差器
  Ptr<detail::BundleAdjusterBase> adjuster;
  if (ba_cost_func == "reproj")
    adjuster = makePtr<detail::BundleAdjusterReproj>();
  else if (ba_cost_func == "ray")
    adjuster = makePtr<detail::BundleAdjusterRay>();
  else if (ba_cost_func == "affine")
    adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
  else if (ba_cost_func == "no") 
    adjuster = makePtr<NoBundleAdjuster>();
  else {
    cout << "Unknown bundle adjustment cost function: '"
         << ba_cost_func
         << "'.\n";
    assert(false);
  }
  
  // 设置置信度阈值
  adjuster->setConfThresh(conf_thresh);
  
  // 创建优化掩码，决定哪些参数需要优化
  Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
  if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1; // 优化fx
  if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1; // 优化fy
  if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1; // 优化cx
  if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1; // 优化cy
  if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1; // 优化s
  
  // 设置优化掩码
  adjuster->setRefinementMask(refine_mask);
  
  // 执行光束法平差优化相机参数
  if (!(*adjuster)(features, pairwise_matches, camera_params_vector_)) {
    cout << "Camera parameters adjusting failed.\n";
    assert(false);
  }

  // 波浪校正，消除累积误差
  vector<Mat> rmats;
  for (size_t i = 0; i < camera_params_vector_.size(); ++i)
    rmats.push_back(camera_params_vector_[i].R.clone());
  waveCorrect(rmats, wave_correct);
  
  // 更新相机旋转矩阵并输出结果
  for (size_t i = 0; i < camera_params_vector_.size(); ++i) {
    camera_params_vector_[i].R = rmats[i];
    LOGLN("Initial camera intrinsics #"
              << i + 1 << ":\nK:\n"
              << camera_params_vector_[i].K()
              << "\nR:\n" << camera_params_vector_[i].R);
  }
}


/**
 * 初始化图像变形器
 * 创建适合的图像变形器并计算重映射映射表
 */
void StitchingParamGenerator::InitWarper() {
  // 存储所有相机的焦距
  vector<double> focals;
  float median_focal_length;
  
  // 初始化重投影映射表
  reproj_xmap_vector_ = vector<UMat>(num_img_);
  reproj_ymap_vector_ = vector<UMat>(num_img_);

  // 输出相机参数并收集焦距
  for (size_t i = 0; i < camera_params_vector_.size(); ++i) {
    LOGLN("Camera #" << i + 1 << ":\nK:\n" << camera_params_vector_[i].K()
                     << "\nR:\n" << camera_params_vector_[i].R);
    focals.push_back(camera_params_vector_[i].focal);
  }
  
  // 计算中值焦距
  sort(focals.begin(), focals.end());
  if (focals.size() % 2 == 1)
    median_focal_length = static_cast<float>(focals[focals.size() / 2]);
  else
    median_focal_length =
        static_cast<float>(focals[focals.size() / 2 - 1] +
                           focals[focals.size() / 2]) * 0.5f;

  // 创建变形器
  Ptr<WarperCreator> warper_creator;
  
  // 使用CPU版本的变形器（OpenCL会通过UMat自动加速）
  LOGLN("Using warper: " << warp_type);
  if (warp_type == "plane")
    warper_creator = makePtr<cv::PlaneWarper>();
  else if (warp_type == "affine")
    warper_creator = makePtr<cv::AffineWarper>();
  else if (warp_type == "cylindrical")
    warper_creator = makePtr<cv::CylindricalWarper>();
  else if (warp_type == "spherical")
    warper_creator = makePtr<cv::SphericalWarper>();
  else if (warp_type == "fisheye")
    warper_creator = makePtr<cv::FisheyeWarper>();
  else if (warp_type == "stereographic")
    warper_creator = makePtr<cv::StereographicWarper>();
  else if (warp_type == "compressedPlaneA2B1")
    warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
  else if (warp_type == "compressedPlaneA1.5B1")
    warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
  else if (warp_type == "compressedPlanePortraitA2B1")
    warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
  else if (warp_type == "compressedPlanePortraitA1.5B1")
    warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
  else if (warp_type == "paniniA2B1")
    warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
  else if (warp_type == "paniniA1.5B1")
    warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
  else if (warp_type == "paniniPortraitA2B1")
    warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
  else if (warp_type == "paniniPortraitA1.5B1")
    warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
  else if (warp_type == "mercator")
    warper_creator = makePtr<cv::MercatorWarper>();
  else if (warp_type == "transverseMercator")
    warper_creator = makePtr<cv::TransverseMercatorWarper>();
  
  // 检查变形器是否创建成功
  if (!warper_creator) {
    cout << "Can't create the following warper '" << warp_type << "'\n";
    assert(false);
  }
  
  // 使用中值焦距创建旋转变形器
  rotation_warper_ = warper_creator->create(static_cast<float>(median_focal_length));
  LOGLN("warped_image_scale: " << median_focal_length);

  // 存储每个图像的边界框
  Rect rect;
  vector<cv::Point> image_point_vect(num_img_);

  // 为每个图像构建重映射映射表
  for (int img_idx = 0; img_idx < num_img_; ++img_idx) {
    Mat_<float> K;
    camera_params_vector_[img_idx].K().convertTo(K, CV_32F);
    
    // 构建重映射映射表
    rect = rotation_warper_->buildMaps(image_size_vector_[img_idx], K,
                                       camera_params_vector_[img_idx].R,
                                       reproj_xmap_vector_[img_idx],
                                       reproj_ymap_vector_[img_idx]);
    
    // 记录图像的左上角点
    Point point(rect.x, rect.y);
    image_point_vect[img_idx] = point;
  }

  // 准备图像掩码
  for (int img_idx = 0; img_idx < num_img_; ++img_idx) {
    // 创建全白掩码
    mask_vector_[img_idx].create(image_vector_[img_idx].size(), CV_8U);
    mask_vector_[img_idx].setTo(Scalar::all(255));
    
    // 对掩码应用重映射
    remap(mask_vector_[img_idx],
          mask_warped_vector_[img_idx],
          reproj_xmap_vector_[img_idx],
          reproj_ymap_vector_[img_idx],
          INTER_NEAREST);
    
    // 记录变形后图像的尺寸
    image_warped_size_vector_[img_idx] = mask_warped_vector_[img_idx].size();
  }

  // 创建延时摄影器和融合器
  timelapser_ = Timelapser::createDefault(timelapse_type);
  blender_ = Blender::createDefault(Blender::NO);
  
  // 初始化延时摄影器和融合器
  timelapser_->initialize(image_point_vect, image_size_vector_);
  blender_->prepare(image_point_vect, image_size_vector_);

  // 存储投影图像的ROI区域
  vector<cv::Rect> projected_image_roi_vect = vector<cv::Rect>(num_img_);

  // 更新角点和尺寸
  Point roi_tl_bias(999999, 999999); // 初始化为一个很大的值
  
  // 计算所有图像的ROI区域和左上角偏移
  for (int i = 0; i < num_img_; ++i) {
    Size sz = image_vector_[i].size();
    Mat K;
    camera_params_vector_[i].K().convertTo(K, CV_32F);
    
    // 计算变形后的ROI区域
    Rect roi = rotation_warper_->warpRoi(sz, K, camera_params_vector_[i].R);
    cout << "roi" << roi << endl;
    
    // 更新左上角偏移，取所有图像中最小的x和y
    roi_tl_bias.x = min(roi.tl().x, roi_tl_bias.x);
    roi_tl_bias.y = min(roi.tl().y, roi_tl_bias.y);
    
    projected_image_roi_vect[i] = roi;
  }
  
  // 计算完整拼接图像的大小和Y方向范围
  full_image_size_ = Point(0, 0);
  Point y_range = Point(-9999999, 999999);
  
  for (int i = 0; i < num_img_; ++i) {
    // 应用左上角偏移，使所有图像的ROI区域从(0,0)开始
    projected_image_roi_vect[i] -= roi_tl_bias;
    Point tl = projected_image_roi_vect[i].tl();
    Point br = projected_image_roi_vect[i].br();

    // 更新完整图像的大小
    full_image_size_.x = max(br.x, full_image_size_.x);
    full_image_size_.y = max(br.y, full_image_size_.y);
    
    // 更新Y方向范围
    y_range.x = max(y_range.x, tl.y);
    y_range.y = min(y_range.y, br.y);
  }
  
  // 正确计算全局Y范围
  int min_y = INT_MAX, max_y = INT_MIN;
  for (int i = 0; i < num_img_; ++i) {
    Point tl = projected_image_roi_vect[i].tl();
    Point br = projected_image_roi_vect[i].br();
    min_y = min(min_y, tl.y);
    max_y = max(max_y, br.y);
  }
  
  // 调整每个图像的ROI区域
  for (int i = 0; i < num_img_; ++i) {
    Rect rect = projected_image_roi_vect[i];
    // 调整Y坐标，使所有图像的Y坐标从0开始
    rect.y = max(0, rect.y - min_y);
    // 确保高度不超过合理范围
    rect.height = max(0, min(rect.height, image_vector_[0].rows));
    // 确保宽度不超过合理范围
    rect.width = max(0, min(rect.width, image_vector_[0].cols));
    
    projected_image_roi_vect[i] = rect;
    projected_image_roi_vect_refined_[i] = rect;
  }

  // 精细化ROI区域，处理潜在的负坐标
  for (int i = 0; i < num_img_ - 1; ++i) {
    Rect rect_left = projected_image_roi_vect_refined_[i];
    
    // 计算两个图像之间的重叠区域的一半作为偏移
    int offset = (projected_image_roi_vect[i].br().x -
                  projected_image_roi_vect[i + 1].tl().x) / 2;
    
    // 确保偏移不会使宽度变为负数
    offset = max(0, min(offset, min(rect_left.width, projected_image_roi_vect[i + 1].width / 2)));
    
    // 调整左侧图像的宽度
    rect_left.width -= offset;
    
    // 调整右侧图像的宽度和X坐标
    Rect rect_right = projected_image_roi_vect[i + 1];
    rect_right.width -= offset;
    rect_right.x = max(0, offset); // 确保X坐标非负
    
    // 更新精细化后的ROI区域
    projected_image_roi_vect_refined_[i] = rect_left;
    projected_image_roi_vect_refined_[i + 1] = rect_right;
  }

  // 最终验证：确保所有ROI区域都在合理范围内
  for (int i = 0; i < num_img_; ++i) {
    Rect& rect = projected_image_roi_vect_refined_[i];
    // 确保x和y非负
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    // 确保宽度和高度为正且在合理范围内
    rect.width = max(0, min(rect.width, image_vector_[0].cols));
    rect.height = max(0, min(rect.height, image_vector_[0].rows));
  }
}


/**
 * 初始化去畸变映射表
 * 从相机配置文件中读取参数并生成去畸变映射表
 */
void StitchingParamGenerator::InitUndistortMap() {
  // 存储每个相机的焦距
  std::vector<double> cam_focal_vector(num_img_);

  // 存储相机参数
  std::vector<cv::UMat> r_vector(num_img_);     // 旋转矩阵
  std::vector<cv::UMat> k_vector(num_img_);     // 内参矩阵
  std::vector<std::vector<double>> d_vector(num_img_); // 畸变系数

  // 初始化去畸变映射表
  undist_xmap_vector_ = std::vector<cv::UMat>(num_img_);
  undist_ymap_vector_ = std::vector<cv::UMat>(num_img_);

  // 从配置文件中读取相机参数
  for (size_t i = 0; i < num_img_; i++) {
    // 打开相机配置文件
    cv::FileStorage fs_read(
        params_dir_ + "/camchain_" + std::to_string(i) + ".yaml",
        cv::FileStorage::READ);
    
    // 检查文件是否打开成功
    if (!fs_read.isOpened()) {
      fprintf(stderr, "%s:%d:loadParams falied. 'camera.yml' does not exist\n",
              __FILE__, __LINE__);
      return;
    }
    
    // 读取相机参数
    cv::Mat R, K;
    fs_read["KMat"] >> K;  // 内参矩阵
    cv::Size calibration_size = ReadCalibrationImageSize(fs_read);
    const cv::Size target_size = image_size_vector_[i];
    cv::Mat scaled_k = K.clone();
    if (calibration_size.width > 0 && calibration_size.height > 0 &&
        target_size.width > 0 && target_size.height > 0 &&
        calibration_size != target_size) {
      const double scale_x =
          static_cast<double>(target_size.width) / calibration_size.width;
      const double scale_y =
          static_cast<double>(target_size.height) / calibration_size.height;
      scaled_k.at<double>(0, 0) *= scale_x;
      scaled_k.at<double>(1, 1) *= scale_y;
      scaled_k.at<double>(0, 2) *= scale_x;
      scaled_k.at<double>(1, 2) *= scale_y;

      std::cout << "[StitchingParamGenerator] Scaling camchain_" << i
                << " intrinsics from " << calibration_size.width << "x"
                << calibration_size.height << " to " << target_size.width
                << "x" << target_size.height << std::endl;
    }
    scaled_k.copyTo(k_vector[i]);
    fs_read["D"] >> d_vector[i];  // 畸变系数
    fs_read["RMat"] >> R;  // 旋转矩阵
    R.copyTo(r_vector[i]);
    fs_read["focal"] >> cam_focal_vector[i];  // 焦距
  }

  // 为每个相机生成去畸变映射表
  for (size_t i = 0; i < num_img_; i++) {
    cv::UMat K;     // 内参矩阵
    cv::UMat R;     // 旋转矩阵
    cv::UMat NONE;  // 空矩阵
    
    // 转换内参矩阵为32位浮点数
    k_vector[i].convertTo(K, CV_32F);
    // 创建单位旋转矩阵
    cv::UMat::eye(3, 3, CV_32F).convertTo(R, CV_32F);

    // 生成去畸变映射表
    cv::initUndistortRectifyMap(
        K,              // 内参矩阵
        d_vector[i],    // 畸变系数
        R,              // 旋转矩阵
        NONE,           // 新的相机矩阵（使用默认）
        image_size_vector_[i],  // 使用输入视频实际分辨率
        CV_32FC1,       // 映射表类型
        undist_xmap_vector_[i],  // x方向映射表
        undist_ymap_vector_[i]); // y方向映射表
  }

}

cv::Size StitchingParamGenerator::ReadCalibrationImageSize(
    cv::FileStorage& fs_read) {
  int width = 0;
  int height = 0;

  fs_read["width"] >> width;
  fs_read["height"] >> height;
  if (width > 0 && height > 0) {
    return cv::Size(width, height);
  }

  cv::FileNode resolution = fs_read["resolution"];
  if (!resolution.empty() && resolution.isSeq() && resolution.size() >= 2) {
    width = static_cast<int>(resolution[0]);
    height = static_cast<int>(resolution[1]);
  }

  return cv::Size(width, height);
}

/**
 * 获取重投影参数
 * @param undist_xmap_vector 输出的去畸变x方向映射表
 * @param undist_ymap_vector 输出的去畸变y方向映射表
 * @param reproj_xmap_vector 输出的重投影x方向映射表
 * @param reproj_ymap_vector 输出的重投影y方向映射表
 * @param projected_image_roi_vect_refined 输出的精细化后的投影图像ROI区域
 */
void StitchingParamGenerator::GetReprojParams(
    vector<cv::UMat>& undist_xmap_vector,
    vector<cv::UMat>& undist_ymap_vector,
    vector<cv::UMat>& reproj_xmap_vector,
    vector<cv::UMat>& reproj_ymap_vector,
    vector<cv::Rect>& projected_image_roi_vect_refined) {
  // 复制去畸变映射表
  undist_xmap_vector = undist_xmap_vector_;
  undist_ymap_vector = undist_ymap_vector_;
  
  // 复制重投影映射表
  reproj_xmap_vector = reproj_xmap_vector_;
  reproj_ymap_vector = reproj_ymap_vector_;
  
  // 复制精细化后的ROI区域
  projected_image_roi_vect_refined = projected_image_roi_vect_refined_;

}
