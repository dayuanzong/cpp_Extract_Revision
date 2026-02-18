#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <algorithm>
#include <iostream>

struct Face {
    float x1, y1, x2, y2, score;
};

class S3FDExtractor {
public:
    S3FDExtractor(const std::wstring& model_path, int device_id = 0);
    std::vector<Face> Detect(const cv::Mat& img, float threshold = 0.5f);

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;
    
    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    std::vector<int64_t> input_node_dims;
    
    // Constants
    const float mean[3] = {104.0f, 117.0f, 123.0f}; // BGR
    
    void PostProcess(const std::vector<std::vector<float>>& outputs, 
                     int width, int height, float scale, 
                     std::vector<Face>& faces, float threshold);
                     
    void NMS(std::vector<Face>& faces, float threshold);
};
