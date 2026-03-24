//
// Created by s1nh.org on 2020/11/14.
//

#ifndef IMAGE_STITCHING_STITCHING_PARAM_GENERATER_H
#define IMAGE_STITCHING_STITCHING_PARAM_GENERATER_H

#include "opencv2/opencv.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"

using namespace std;

/**
 * 图像拼接参数生成器类
 * 负责计算和生成图像拼接所需的各种参数，包括相机参数、重映射映射表和ROI区域
 */
class StitchingParamGenerator {
 public:
    /**
     * 构造函数
     * @param image_vector 输入图像向量，包含要拼接的所有图像
     */
    explicit StitchingParamGenerator(const vector<cv::Mat>& image_vector);

    /**
     * 获取重投影参数
     * @param undist_xmap_vector 输出的去畸变x方向映射表
     * @param undist_ymap_vector 输出的去畸变y方向映射表
     * @param reproj_xmap_vector 输出的重投影x方向映射表
     * @param reproj_ymap_vector 输出的重投影y方向映射表
     * @param projected_image_roi_vect_refined 输出的精细化后的投影图像ROI区域
     */
    void GetReprojParams(vector<cv::UMat>& undist_xmap_vector,
                         vector<cv::UMat>& undist_ymap_vector,
                         vector<cv::UMat>& reproj_xmap_vector,
                         vector<cv::UMat>& reproj_ymap_vector,
                         vector<cv::Rect>& projected_image_roi_vect_refined);

    /**
     * 初始化相机参数
     * 包括特征点检测、匹配、相机参数估计和光束法平差
     */
    void InitCameraParam();

    /**
     * 初始化图像变形器
     * 创建适合的图像变形器并计算重映射映射表
     */
    void InitWarper();

    /**
     * 初始化去畸变映射表
     * 从相机配置文件中读取参数并生成去畸变映射表
     */
    void InitUndistortMap();

    cv::Size ReadCalibrationImageSize(cv::FileStorage& fs_read);


 private:
    // 默认命令行参数
    vector<cv::String> img_names;        // 图像文件名
    bool try_opencl = true;              // 启用OpenCL加速
    float conf_thresh = 1.f;             // 置信度阈值
    float match_conf = 0.6f;             // 匹配置信度
    string matcher_type = "homography";  // 匹配器类型
    string estimator_type = "homography"; // 估计器类型
    string ba_cost_func = "reproj";      // 光束法平差代价函数
    string ba_refine_mask = "xxxxx";     // 光束法平差优化掩码
    cv::detail::WaveCorrectKind wave_correct = cv::detail::WAVE_CORRECT_HORIZ; // 波浪校正类型
    bool save_graph = false;              // 是否保存匹配图
    string save_graph_to;                // 匹配图保存路径
    string warp_type = "spherical";      // 变形类型
    int expos_comp_type = cv::detail::ExposureCompensator::GAIN_BLOCKS; // 曝光补偿类型
    int expos_comp_nr_feeds = 1;         // 曝光补偿馈送数
    int expos_comp_nr_filtering = 2;     // 曝光补偿滤波次数
    int expos_comp_block_size = 32;      // 曝光补偿块大小
    string seam_find_type = "no";        // 接缝查找类型
    int blend_type = cv::detail::Blender::MULTI_BAND; // 融合类型
    int timelapse_type = cv::detail::Timelapser::AS_IS; // 延时摄影类型
    float blend_strength = 5;            // 融合强度
    string result_name = "../results/result.jpg"; // 结果文件名
    bool timelapse = true;               // 是否使用延时摄影
    int range_width = -1;                // 范围宽度
    string params_dir_ = "../params";    // 相机参数目录

    // 变量
    size_t num_img_;                     // 图像数量
    cv::Point full_image_size_;          // 完整拼接图像的大小

    vector<cv::Mat> image_vector_;       // 输入图像向量
    vector<cv::UMat> mask_vector_;       // 图像掩码向量
    vector<cv::UMat> mask_warped_vector_; // 变形后的掩码向量
    vector<cv::Size> image_size_vector_;  // 图像尺寸向量
    vector<cv::Size> image_warped_size_vector_; // 变形后图像尺寸向量
    vector<cv::UMat> reproj_xmap_vector_; // 重投影x方向映射表
    vector<cv::UMat> reproj_ymap_vector_; // 重投影y方向映射表
    vector<cv::UMat> undist_xmap_vector_; // 去畸变x方向映射表
    vector<cv::UMat> undist_ymap_vector_; // 去畸变y方向映射表

    vector<cv::detail::CameraParams> camera_params_vector_; // 相机参数向量
    vector<cv::Rect> projected_image_roi_vect_refined_; // 精细化后的投影图像ROI区域
    cv::Ptr<cv::detail::RotationWarper> rotation_warper_; // 旋转变形器
    cv::Ptr<cv::detail::Timelapser> timelapser_; // 延时摄影器
    cv::Ptr<cv::detail::Blender> blender_; // 图像融合器
};


#endif //IMAGE_STITCHING_STITCHING_PARAM_GENERATER_H
