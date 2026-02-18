#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>

using Point2f = cv::Point2f;

class FANExtractor {
public:
    FANExtractor(const std::wstring& model_path, int device_id = 0);
    std::vector<Point2f> Extract(const cv::Mat& img, const cv::Rect2f& face_rect, bool multi_sample = false);

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;
    
    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    
    // Constants
    const float input_size = 256.0f;
    const float scale_factor = 1.0f / 255.0f;
    
    cv::Mat Crop(const cv::Mat& img, cv::Point2f center, float scale);
    std::vector<Point2f> ExtractSingle(const cv::Mat& img, cv::Point2f center, float scale);
    std::vector<Point2f> PostProcess(const std::vector<float>& heatmap, cv::Point2f center, float scale);
};
