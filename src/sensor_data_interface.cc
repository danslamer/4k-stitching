//
// Created by s1nh.org on 11/11/20.
// https://zhuanlan.zhihu.com/p/38136322
//

#include "sensor_data_interface.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

namespace {

std::string GetEnvOrDefault(const char* name, const std::string& default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0')
    return default_value;
  return value;
}

std::vector<std::string> CollectVideoFileNames(const std::string& video_dir) {
  std::vector<std::string> video_file_names;
  std::vector<cv::String> video_paths;
  cv::glob(video_dir + "*.mp4", video_paths, false);
  std::vector<cv::String> uppercase_video_paths;
  cv::glob(video_dir + "*.MP4", uppercase_video_paths, false);
  video_paths.insert(video_paths.end(),
                     uppercase_video_paths.begin(),
                     uppercase_video_paths.end());

  for (size_t i = 0; i < video_paths.size(); ++i) {
    std::string file_path = video_paths[i];
    const size_t slash_pos = file_path.find_last_of("/\\");
    if (slash_pos == std::string::npos)
      video_file_names.push_back(file_path);
    else
      video_file_names.push_back(file_path.substr(slash_pos + 1));
  }

  std::sort(video_file_names.begin(), video_file_names.end());
  return video_file_names;
}

}  // namespace

SensorDataInterface::SensorDataInterface()
    : max_queue_length_(2) {
  num_img_ = 0;
}

//void SensorDataInterface::InitExampleImages() {
//  std::string img_dir = "../datasets/cam01/pic_raw/";
//  std::vector<std::string> img_file_name = {"0.jpg",
//                                            "1.jpg",
//                                            "2.jpg",
//                                            "3.jpg",
//                                            "4.jpg"};
//
//  num_img_ = img_file_name.size();
//  image_queue_vector_ = std::vector<std::queue<cv::UMat>>(num_img_);
//
//  for (int i = 0; i < img_file_name.size(); ++i) {
//    std::string file_name = img_dir + img_file_name[i];
//    cv::UMat _;
//    cv::imread(file_name, 1).copyTo(_);
//    image_queue_vector_[i].push(_);
//  }
//}


void SensorDataInterface::InitVideoCapture(size_t& num_img) {
  std::string video_dir = GetEnvOrDefault("STITCH_VIDEO_DIR", "../datasets/4k-test/");
  if (!video_dir.empty() && video_dir.back() != '/' && video_dir.back() != '\\') {
    video_dir += "/";
  }
  std::vector<std::string> video_file_name = CollectVideoFileNames(video_dir);
  if (video_file_name.size() != 4) {
    std::cerr << "[SensorDataInterface] Expected 4 video files in " << video_dir
              << ", but found " << video_file_name.size() << "." << std::endl;
  }

  num_img_ = video_file_name.size();
  num_img = num_img_;
  image_queue_vector_ = std::vector<std::queue<cv::UMat>>(num_img_);
  image_queue_mutex_vector_ = std::vector<std::mutex>(num_img_);
  frame_size_vector_ = std::vector<cv::Size>(num_img_);

  // Init video capture.
  for (int i = 0; i < num_img_; ++i) {
    std::string file_name = video_dir + video_file_name[i];

    cv::VideoCapture capture(file_name);
    if (!capture.isOpened()) {
      std::cerr << "[SensorDataInterface] Failed to open video: "
                << file_name << std::endl;
    }
    video_capture_vector_.push_back(capture);

    cv::UMat frame;
    capture.read(frame);
    frame_size_vector_[i] = frame.size();
    std::cout << "[SensorDataInterface] Loaded video " << file_name
              << " with frame size " << frame.cols << "x" << frame.rows
              << std::endl;
    image_queue_vector_[i].push(frame);

  }
}


void SensorDataInterface::RecordVideos() {

  size_t frame_idx = 0;
  while (true) {
    for (int i = 0; i < num_img_; ++i) {
      cv::UMat frame;
      video_capture_vector_[i].read(frame);
      if (frame.rows > 0) {
        image_queue_mutex_vector_[i].lock();
        image_queue_vector_[i].push(frame);
        if (image_queue_vector_[i].size() > max_queue_length_) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        image_queue_mutex_vector_[i].unlock();
      } else {
        break;
      }
    }
    frame_idx++;
//    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  }
}


void
SensorDataInterface::get_image_vector(
    std::vector<cv::UMat>& image_vector,
    std::vector<std::mutex>& image_mutex_vector) {

  for (size_t i = 0; i < num_img_; ++i) {
    cv::Mat img_undistort;
    cv::Mat img_cylindrical;

    while (image_queue_vector_[i].empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    image_queue_mutex_vector_[i].lock();
    image_mutex_vector[i].lock();
    image_vector[i] = image_queue_vector_[i].front();
    image_queue_vector_[i].pop();
    image_mutex_vector[i].unlock();
    image_queue_mutex_vector_[i].unlock();
  }
}

const std::vector<cv::Size>& SensorDataInterface::frame_size_vector() const {
  return frame_size_vector_;
}
