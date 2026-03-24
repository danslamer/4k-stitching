//
// Created by s1nh.org on 2020/12/1.
//

#ifndef IMAGE_STITCHING_IMAGE_STITCHER_H
#define IMAGE_STITCHING_IMAGE_STITCHER_H

#include "opencv2/opencv.hpp"

using namespace std;

/**
 * 图像拼接器类
 * 负责将多个图像进行拼接，包括图像变形、融合和合成
 */
class ImageStitcher {


 public:
    /**
     * 设置拼接参数
     * @param blend_width 融合宽度
     * @param undist_xmap_vector 去畸变x方向映射表
     * @param undist_ymap_vector 去畸变y方向映射表
     * @param reproj_xmap_vector 重投影x方向映射表
     * @param reproj_ymap_vector 重投影y方向映射表
     * @param projected_image_roi_vect_refined 精细化后的投影图像ROI区域
     */
    void SetParams(
        const int& blend_width,
        vector<cv::UMat>& undist_xmap_vector,
        vector<cv::UMat>& undist_ymap_vector,
        vector<cv::UMat>& reproj_xmap_vector,
        vector<cv::UMat>& reproj_ymap_vector,
        vector<cv::Rect>& projected_image_roi_vect_refined);

    /**
     * 对图像进行变形和融合
     * @param img_idx 图像索引
     * @param fusion_pixel 融合像素数
     * @param image_vector 输入图像向量
     * @param image_mutex_vector 图像互斥锁向量
     * @param images_warped_with_roi_vector 带ROI的变形图像向量
     * @param image_concat_umat 拼接后的图像
     */
    void WarpImages(
        const int& img_idx,
        const int& fusion_pixel,
        const vector<cv::UMat>& image_vector,
        vector<mutex>& image_mutex_vector,
        vector<cv::UMat>& images_warped_with_roi_vector,
        cv::UMat& image_concat_umat);

    /**
     * 简单图像融合
     * @param fusion_pixel 融合像素数
     * @param img_vect 图像向量
     */
    void SimpleImageBlender(
        const size_t& fusion_pixel,
        vector<cv::UMat>& img_vect);

 private:
    size_t num_img_;  // 图像数量

//    cv::UMat warp_tmp_l_;  // 临时变形图像
    vector<cv::UMat> reproj_xmap_vector_;  // 重投影x方向映射表
    vector<cv::UMat> reproj_ymap_vector_;  // 重投影y方向映射表
    vector<cv::UMat> undist_xmap_vector_;  // 去畸变x方向映射表
    vector<cv::UMat> undist_ymap_vector_;  // 去畸变y方向映射表
    vector<cv::UMat> final_xmap_vector_;   // 最终x方向映射表（合并去畸变和重投影）
    vector<cv::UMat> final_ymap_vector_;   // 最终y方向映射表（合并去畸变和重投影）
    vector<cv::UMat> tmp_umat_vect_;       // 临时UMat向量
//    vector<cv::UMat> wrap_vec_;            // 变形向量
    vector<mutex> warp_mutex_vector_;       // 变形互斥锁向量
    vector<cv::Rect> roi_vect_;             // ROI区域向量
    vector<cv::UMat> weightMap_;            // 权重图向量

    /**
     * 创建权重图
     * @param height 高度
     * @param width 宽度
     */
    void CreateWeightMap(const int& height, const int& width);

};


#endif //IMAGE_STITCHING_IMAGE_STITCHER_H