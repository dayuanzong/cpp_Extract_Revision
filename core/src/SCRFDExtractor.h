#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include "S3FDExtractor.h" // For Face struct

class SCRFDExtractor {
public:
    SCRFDExtractor(const std::wstring& model_path, int device_id = 0, int input_size_value = 640);
    std::vector<Face> Detect(const cv::Mat& img, float threshold = 0.5f);
    int GetInputSize() const { return input_size; }

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;

    // SCRFD specific
    int input_size = 640;
    const float mean[3] = {127.5f, 127.5f, 127.5f};
    const float std[3] = {128.0f, 128.0f, 128.0f};
    
    // Anchors/Strides
    const std::vector<int> strides = {8, 16, 32};
    
    void GenerateAnchors(int height, int width, std::vector<std::vector<float>>& anchors);
    
    struct OutputTensorInfo {
        float* data;
        std::vector<int64_t> dims;
    };

    void PostProcess(const std::vector<OutputTensorInfo>& outputs, 
                     const std::vector<std::vector<float>>& anchors,
                     float scale, std::vector<Face>& faces, float threshold);
    void NMS(std::vector<Face>& faces, float threshold);
};
