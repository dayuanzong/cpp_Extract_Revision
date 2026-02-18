#include "InsightFaceLandmark.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cctype>

InsightFaceLandmark::InsightFaceLandmark(const std::wstring& model_path, int device_id)
    : env(ORT_LOGGING_LEVEL_WARNING, "InsightFaceLandmark"), model_path_str(model_path) {
    
    Ort::SessionOptions session_options;
    
    if (device_id >= 0) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = device_id;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "InsightFaceLandmark: CUDA Provider enabled on device " << device_id << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "InsightFaceLandmark: Failed to enable CUDA Provider: " << e.what() << std::endl;
            std::cerr << "InsightFaceLandmark: Falling back to CPU." << std::endl;
        }
    } else {
        std::cout << "InsightFaceLandmark: CPU Mode requested." << std::endl;
    }

    session = Ort::Session(env, model_path.c_str(), session_options);
    
    // Get input/output names
    size_t num_input_nodes = session.GetInputCount();
    input_names_str.reserve(num_input_nodes);
    input_node_names.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        input_names_str.push_back(input_name.get());
        input_node_names.push_back(input_names_str.back().c_str());
    }

    size_t num_output_nodes = session.GetOutputCount();
    output_names_str.reserve(num_output_nodes);
    output_node_names.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        output_names_str.push_back(output_name.get());
        output_node_names.push_back(output_names_str.back().c_str());
    }

    if (num_input_nodes > 0) {
        try {
            auto in_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            auto in_shape = in_info.GetShape();
            if (in_shape.size() >= 4) {
                int64_t h = in_shape[in_shape.size() - 2];
                int64_t w = in_shape[in_shape.size() - 1];
                if (h > 0) input_h = (int)h;
                if (w > 0) input_w = (int)w;
            } else if (in_shape.size() >= 3) {
                int64_t h = in_shape[in_shape.size() - 2];
                int64_t w = in_shape[in_shape.size() - 1];
                if (h > 0) input_h = (int)h;
                if (w > 0) input_w = (int)w;
            }
        } catch (...) {
        }
    }
    
    // Initialize configurations
    InitializeConfigs();
    
    // Detect model type
    DetectModelType();
    
    std::cout << "InsightFaceLandmark: Model type detected as " 
              << (model_type == ModelType::MODEL_1K3D68 ? "1K3D68" : 
                  model_type == ModelType::MODEL_2D106DET ? "2D106DET" : "UNKNOWN") 
              << std::endl;
}

void InsightFaceLandmark::InitializeConfigs() {
    // 1k3d68 configuration
    // Note: Keep crop_factor same as 2d106det for now (1.75f)
    // IMPORTANT: Multi-sampling temporarily disabled to debug crop issue
    configs[ModelType::MODEL_1K3D68] = {
        ModelType::MODEL_1K3D68,
        {
            1.75f,  // crop_factor - same as original, to be optimized through experiments
            true,  // use_padding
            cv::Scalar(127, 127, 127)  // pad_value
        },
        {
            NormMode::AUTO,  // norm_mode - changed back to AUTO to match original behavior
            0.0f,   // mean
            1.0f    // std
        },
        {
            false,  // enabled - DISABLED for now to debug
            1,      // sample_count
            0.0f    // offset_pixels
        }
    };
    
    // 2d106det configuration (keep existing behavior)
    configs[ModelType::MODEL_2D106DET] = {
        ModelType::MODEL_2D106DET,
        {
            1.75f,  // crop_factor - keep current value
            true,   // use_padding
            cv::Scalar(127, 127, 127)  // pad_value
        },
        {
            NormMode::AUTO,  // norm_mode - keep auto-select behavior
            0.0f,   // mean
            1.0f    // std
        },
        {
            false,  // enabled - no multi-sampling for 2d106det
            1,      // sample_count
            0.0f    // offset_pixels
        }
    };
}

void InsightFaceLandmark::DetectModelType() {
    // Method 1: Detect by filename
    std::wstring path_lower = model_path_str;
    std::transform(path_lower.begin(), path_lower.end(), path_lower.begin(), ::towlower);
    
    if (path_lower.find(L"1k3d68") != std::wstring::npos) {
        model_type = ModelType::MODEL_1K3D68;
        return;
    }
    
    if (path_lower.find(L"2d106det") != std::wstring::npos) {
        model_type = ModelType::MODEL_2D106DET;
        return;
    }
    
    // Method 2: Detect by output shape (will be determined after first inference)
    // For now, default to 2D106DET to maintain backward compatibility
    model_type = ModelType::MODEL_2D106DET;
    std::cout << "InsightFaceLandmark: Model type not detected from filename, defaulting to 2D106DET" << std::endl;
}

const ModelConfig& InsightFaceLandmark::GetConfig() const {
    auto it = configs.find(model_type);
    if (it != configs.end()) {
        return it->second;
    }
    // Default to 2D106DET config for backward compatibility
    return configs.at(ModelType::MODEL_2D106DET);
}

cv::Mat InsightFaceLandmark::Crop(const cv::Mat& img, const cv::Rect2f& rect, cv::Mat& M_inv, const cv::Point2f& center_offset) {
    cv::Point2f center(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
    
    // Apply center offset for multi-sampling
    center.x += center_offset.x;
    center.y += center_offset.y;
    
    float size = std::max(rect.width, rect.height);
    if (size < 1.0f) size = 1.0f;

    // Get crop factor from configuration
    const auto& config = GetConfig();
    float crop_factor = config.crop.crop_factor;

    cv::Mat M = cv::Mat::zeros(2, 3, CV_32F);
    float s = (float)input_w / (size * crop_factor);
    if (s <= 0.0f) s = 1.0f;
    float half_w = (float)input_w * 0.5f;
    float half_h = (float)input_h * 0.5f;
    M.at<float>(0, 0) = s;
    M.at<float>(0, 1) = 0;
    M.at<float>(0, 2) = half_w - center.x * s;
    M.at<float>(1, 0) = 0;
    M.at<float>(1, 1) = s;
    M.at<float>(1, 2) = half_h - center.y * s;
    
    cv::Mat cropped;
    cv::warpAffine(img, cropped, M, cv::Size(input_w, input_h), cv::INTER_LINEAR);
    cv::invertAffineTransform(M, M_inv);
    if (M_inv.type() != CV_64F) {
        M_inv.convertTo(M_inv, CV_64F);
    }
    return cropped;
}

std::vector<cv::Point2f> InsightFaceLandmark::ExtractSingle(const cv::Mat& img, const cv::Rect2f& face_rect, const cv::Point2f& center_offset) {
    cv::Mat M_inv;
    cv::Mat cropped = Crop(img, face_rect, M_inv, center_offset);

    cv::Mat rgb;
    cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3);

    std::vector<int64_t> input_shape = {1, 3, input_h, input_w};
    size_t input_tensor_size = (size_t)1 * (size_t)3 * (size_t)input_h * (size_t)input_w;
    size_t plane = (size_t)input_h * (size_t)input_w;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto decode_output = [&](const Ort::Value& out) -> std::pair<std::vector<float>, std::pair<int, int>> {
        float* output_data = const_cast<Ort::Value&>(out).GetTensorMutableData<float>();
        size_t elem_count = out.GetTensorTypeAndShapeInfo().GetElementCount();
        if (!output_data || elem_count == 0) return {};

        if (elem_count >= 3000 && (elem_count % 3 == 0)) {
            size_t vtx = elem_count / 3;
            if (vtx >= 68) {
                int point_count = 68;
                int coord_dim = 3;
                size_t offset = (vtx - (size_t)point_count) * 3;
                std::vector<float> coords(output_data + offset, output_data + offset + (size_t)point_count * (size_t)coord_dim);
                return {coords, {point_count, coord_dim}};
            }
        }

        int point_count = 0;
        int coord_dim = 0;

        if (elem_count == 212) {
            point_count = 106;
            coord_dim = 2;
        } else if (elem_count == 136) {
            point_count = 68;
            coord_dim = 2;
        } else if (elem_count == 204) {
            point_count = 68;
            coord_dim = 3;
        } else if (elem_count % 3 == 0 && elem_count / 3 == 68) {
            point_count = 68;
            coord_dim = 3;
        } else if (elem_count % 2 == 0) {
            point_count = (int)(elem_count / 2);
            coord_dim = 2;
        } else {
            return {};
        }

        size_t need = (size_t)point_count * (size_t)coord_dim;
        if (need > elem_count) return {};

        std::vector<float> coords(output_data, output_data + need);
        return {coords, {point_count, coord_dim}};
    };

    auto score_landmarks = [&](const std::vector<cv::Point2f>& pts) -> int {
        if (pts.empty()) return 0;
        float x1 = face_rect.x - face_rect.width * 0.5f;
        float y1 = face_rect.y - face_rect.height * 0.5f;
        float x2 = face_rect.x + face_rect.width * 1.5f;
        float y2 = face_rect.y + face_rect.height * 1.5f;
        int ok = 0;
        for (const auto& p : pts) {
            if (p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) ok++;
        }
        return ok;
    };

    std::vector<cv::Point2f> best_pts;
    int best_score = -1;

    // Get preprocessing configuration
    const auto& config = GetConfig();
    const auto& preprocess = config.preprocess;
    
    // Determine which normalization modes to try
    std::vector<int> modes_to_try;
    if (preprocess.norm_mode == NormMode::AUTO) {
        modes_to_try = {0, 1};  // Try both modes
    } else if (preprocess.norm_mode == NormMode::ZERO_ONE) {
        modes_to_try = {0};  // Only ZERO_ONE
    } else {  // MEAN_STD
        modes_to_try = {1};  // Only MEAN_STD
    }

    for (int mode : modes_to_try) {
        float mean = (mode == 0) ? 0.0f : preprocess.mean;
        float stdv = (mode == 0) ? 1.0f : preprocess.std;

        std::vector<float> input_tensor_values(input_tensor_size);
        for (int y = 0; y < input_h; y++) {
            for (int x = 0; x < input_w; x++) {
                cv::Vec3f pixel = rgb.at<cv::Vec3f>(y, x);
                pixel[0] = (pixel[0] - mean) / stdv;
                pixel[1] = (pixel[1] - mean) / stdv;
                pixel[2] = (pixel[2] - mean) / stdv;
                size_t idx = (size_t)y * (size_t)input_w + (size_t)x;
                input_tensor_values[idx] = pixel[0];
                input_tensor_values[plane + idx] = pixel[1];
                input_tensor_values[2 * plane + idx] = pixel[2];
            }
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size());
        if (output_tensors.empty()) continue;

        auto decoded = decode_output(output_tensors[0]);
        if (decoded.first.empty()) continue;
        int point_count = decoded.second.first;
        int coord_dim = decoded.second.second;

        auto pts = PostProcess(decoded.first, point_count, coord_dim, M_inv);
        int sc = score_landmarks(pts);
        if (sc > best_score) {
            best_score = sc;
            best_pts = std::move(pts);
        }
        if (point_count > 0 && sc >= (int)((double)point_count * 0.6)) break;
    }

    return best_pts;
}

std::vector<cv::Point2f> InsightFaceLandmark::ExtractWithMultiSample(const cv::Mat& img, const cv::Rect2f& face_rect) {
    const auto& config = GetConfig();
    const auto& multi_sample = config.multi_sample;
    
    // If multi-sampling is disabled or sample_count <= 1, use single extraction
    if (!multi_sample.enabled || multi_sample.sample_count <= 1) {
        return ExtractSingle(img, face_rect);
    }
    
    // Generate sampling offsets
    std::vector<cv::Point2f> offsets;
    offsets.push_back(cv::Point2f(0, 0));  // Center
    
    if (multi_sample.sample_count >= 5) {
        float off = multi_sample.offset_pixels;
        offsets.push_back(cv::Point2f(-off, 0));   // Left
        offsets.push_back(cv::Point2f(off, 0));    // Right
        offsets.push_back(cv::Point2f(0, -off));   // Up
        offsets.push_back(cv::Point2f(0, off));    // Down
    }
    
    if (multi_sample.sample_count >= 9) {
        float off = multi_sample.offset_pixels;
        offsets.push_back(cv::Point2f(-off, -off)); // Top-left
        offsets.push_back(cv::Point2f(off, -off));  // Top-right
        offsets.push_back(cv::Point2f(-off, off));  // Bottom-left
        offsets.push_back(cv::Point2f(off, off));   // Bottom-right
    }
    
    // Perform multi-sampling and average
    std::vector<cv::Point2f> accum;
    int valid_count = 0;
    
    for (const auto& offset : offsets) {
        auto pts = ExtractSingle(img, face_rect, offset);
        if (pts.empty()) continue;
        
        if (accum.empty()) {
            accum = pts;
        } else {
            for (size_t i = 0; i < pts.size() && i < accum.size(); i++) {
                accum[i].x += pts[i].x;
                accum[i].y += pts[i].y;
            }
        }
        valid_count++;
    }
    
    // Calculate average
    if (valid_count > 0) {
        for (auto& pt : accum) {
            pt.x /= (float)valid_count;
            pt.y /= (float)valid_count;
        }
    }
    
    return accum;
}

std::vector<cv::Point2f> InsightFaceLandmark::Extract(const cv::Mat& img, const cv::Rect2f& face_rect) {
    auto landmarks = ExtractWithMultiSample(img, face_rect);
    
    // Validate output for 1k3d68 model
    if (model_type == ModelType::MODEL_1K3D68 && !landmarks.empty()) {
        auto validation = ValidateOutput(landmarks, face_rect, img.size());
        if (!validation.valid) {
            std::cerr << "InsightFaceLandmark: Validation failed - " << validation.error_message << std::endl;
            if (!validation.invalid_indices.empty()) {
                std::cerr << "  Invalid points: ";
                for (size_t i = 0; i < std::min(validation.invalid_indices.size(), (size_t)5); i++) {
                    std::cerr << validation.invalid_indices[i] << " ";
                }
                if (validation.invalid_indices.size() > 5) {
                    std::cerr << "... (" << validation.invalid_indices.size() << " total)";
                }
                std::cerr << std::endl;
            }
        }
    }
    
    return landmarks;
}

ValidationResult InsightFaceLandmark::ValidateOutput(
    const std::vector<cv::Point2f>& landmarks,
    const cv::Rect2f& face_rect,
    const cv::Size& img_size) const {
    
    ValidationResult result;
    result.valid = true;
    
    // Check landmark count (should be 68 for 1k3d68)
    if (landmarks.size() != 68) {
        result.valid = false;
        result.error_message = "Invalid landmark count: " + std::to_string(landmarks.size()) + " (expected 68)";
        return result;
    }
    
    // Define reasonable bounds (extended face region)
    float x1 = face_rect.x - face_rect.width * 0.5f;
    float y1 = face_rect.y - face_rect.height * 0.5f;
    float x2 = face_rect.x + face_rect.width * 2.5f;
    float y2 = face_rect.y + face_rect.height * 2.5f;
    
    // Also check image bounds
    float img_x1 = -face_rect.width * 0.2f;
    float img_y1 = -face_rect.height * 0.2f;
    float img_x2 = img_size.width + face_rect.width * 0.2f;
    float img_y2 = img_size.height + face_rect.height * 0.2f;
    
    // Check each landmark
    for (size_t i = 0; i < landmarks.size(); i++) {
        const auto& pt = landmarks[i];
        
        // Check for NaN/Inf
        if (std::isnan(pt.x) || std::isnan(pt.y) || 
            std::isinf(pt.x) || std::isinf(pt.y)) {
            result.valid = false;
            result.invalid_indices.push_back((int)i);
            continue;
        }
        
        // Check if point is in reasonable range
        bool in_face_bounds = (pt.x >= x1 && pt.x <= x2 && pt.y >= y1 && pt.y <= y2);
        bool in_img_bounds = (pt.x >= img_x1 && pt.x <= img_x2 && pt.y >= img_y1 && pt.y <= img_y2);
        
        if (!in_face_bounds || !in_img_bounds) {
            result.invalid_indices.push_back((int)i);
        }
    }
    
    // Check if too many invalid points
    float invalid_ratio = (float)result.invalid_indices.size() / (float)landmarks.size();
    if (invalid_ratio > 0.3f) {  // More than 30% invalid
        result.valid = false;
        result.error_message = "Too many invalid points: " + 
                              std::to_string(result.invalid_indices.size()) + 
                              "/" + std::to_string(landmarks.size()) +
                              " (" + std::to_string((int)(invalid_ratio * 100)) + "%)";
    } else if (!result.invalid_indices.empty()) {
        // Some invalid points but not too many - just warning
        result.error_message = "Some points out of bounds: " + 
                              std::to_string(result.invalid_indices.size()) + 
                              "/" + std::to_string(landmarks.size());
    }
    
    return result;
}

std::vector<cv::Point2f> InsightFaceLandmark::PostProcess(const std::vector<float>& coords, int point_count, int coord_dim, const cv::Mat& M_inv) {
    std::vector<cv::Point2f> landmarks;
    if (point_count <= 0 || (coord_dim != 2 && coord_dim != 3)) return landmarks;

    float half_w = (float)input_w * 0.5f;
    float half_h = (float)input_h * 0.5f;
    landmarks.reserve((size_t)point_count);
    for (int i = 0; i < point_count; i++) {
        float x = coords[(size_t)i * (size_t)coord_dim + 0];
        float y = coords[(size_t)i * (size_t)coord_dim + 1];
        float cx = (x + 1.0f) * half_w;
        float cy = (y + 1.0f) * half_h;
        float old_x = cx;
        float old_y = cy;
        if (!M_inv.empty()) {
            double nx = M_inv.at<double>(0, 0) * old_x + M_inv.at<double>(0, 1) * old_y + M_inv.at<double>(0, 2);
            double ny = M_inv.at<double>(1, 0) * old_x + M_inv.at<double>(1, 1) * old_y + M_inv.at<double>(1, 2);
            old_x = (float)nx;
            old_y = (float)ny;
        }
        landmarks.push_back(cv::Point2f(old_x, old_y));
    }

    return landmarks;
}
