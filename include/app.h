//
// Created by s1nh.org on 2020/12/2.
//

#ifndef IMAGE_STITCHING_APP_H
#define IMAGE_STITCHING_APP_H

#include "opencv2/opencv.hpp"

#include "sensor_data_interface.h"
#include "image_stitcher.h"

using namespace std;

/**
 * 应用程序类
 * 负责协调图像采集、参数生成和图像拼接的整个流程
 */
class App {
 public:
    /**
     * 构造函数
     * 初始化传感器数据接口、图像拼接器和相关参数
     */
    App();

    /**
     * 运行图像拼接
     * 无限循环执行图像采集、拼接和输出
     */
    [[noreturn]] void run_stitching();

 private:
    std::size_t num_img_;           // 图像数量
    SensorDataInterface sensorDataInterface_; // 传感器数据接口
    ImageStitcher image_stitcher_;  // 图像拼接器
    vector<cv::Mat> image_vector_;  // 图像向量
    cv::UMat image_concat_umat_;    // 拼接后的图像
    int total_cols_;                // 总列数

};

#endif //IMAGE_STITCHING_APP_H