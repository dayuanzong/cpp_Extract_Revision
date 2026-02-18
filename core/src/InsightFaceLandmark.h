#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <map>

// Model type enumeration
enum class ModelType {
    UNKNOWN,
    MODEL_1K3D68,    // 68 points, 3D coordinates
    MODEL_2D106DET   // 106 points, 2D coordinates
};

// Normalization mode enumeration
enum class NormMode {
    AUTO,           // Auto-select (current behavior)
    ZERO_ONE,       // [0, 1] normalization
    MEAN_STD        // (x-127.5)/128 normalization
};

// Crop configuration
struct CropConfig {
    float crop_factor;      // Crop factor
    bool use_padding;       // Whether to use padding
    cv::Scalar pad_value;   // Padding value
};

// Preprocessing configuration
struct PreprocessConfig {
    NormMode norm_mode;
    float mean;
    float std;
};

// Multi-sampling configuration
struct MultiSampleConfig {
    bool enabled;           // Whether to enable multi-sampling
    int sample_count;       // Number of sampling points (1, 5, 9)
    float offset_pixels;    // Offset in pixels
};

// Model configuration
struct ModelConfig {
    ModelType type;
    CropConfig crop;
    PreprocessConfig preprocess;
    MultiSampleConfig multi_sample;
};

// Validation result
struct ValidationResult {
    bool valid;
    std::string error_message;
    std::vector<int> invalid_indices;
};

class InsightFaceLandmark {
public:
    InsightFaceLandmark(const std::wstring& model_path, int device_id = 0);
    std::vector<cv::Point2f> Extract(const cv::Mat& img, const cv::Rect2f& face_rect);

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;

    int input_w = 192;
    int input_h = 192;
    float input_mean = 0.0f;
    float input_std = 1.0f;

    // Model type and configuration
    ModelType model_type = ModelType::UNKNOWN;
    std::wstring model_path_str;
    std::map<ModelType, ModelConfig> configs;

    // Initialize configurations
    void InitializeConfigs();
    
    // Detect model type
    void DetectModelType();
    
    // Get configuration for current model
    const ModelConfig& GetConfig() const;

    // Single extraction (for multi-sampling)
    std::vector<cv::Point2f> ExtractSingle(const cv::Mat& img, const cv::Rect2f& face_rect, const cv::Point2f& center_offset = cv::Point2f(0, 0));
    
    // Multi-sampling extraction
    std::vector<cv::Point2f> ExtractWithMultiSample(const cv::Mat& img, const cv::Rect2f& face_rect);
    
    // Validate output landmarks
    ValidationResult ValidateOutput(const std::vector<cv::Point2f>& landmarks, const cv::Rect2f& face_rect, const cv::Size& img_size) const;

    cv::Mat Crop(const cv::Mat& img, const cv::Rect2f& rect, cv::Mat& M_inv, const cv::Point2f& center_offset = cv::Point2f(0, 0));
    std::vector<cv::Point2f> PostProcess(const std::vector<float>& coords, int point_count, int coord_dim, const cv::Mat& M_inv);
};
