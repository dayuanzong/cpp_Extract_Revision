#include "SCRFDExtractor.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

SCRFDExtractor::SCRFDExtractor(const std::wstring& model_path, int device_id, int input_size_value)
    : env(ORT_LOGGING_LEVEL_WARNING, "SCRFD") {
    
    Ort::SessionOptions session_options;
    
    if (device_id >= 0) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = device_id;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "SCRFD: CUDA Provider enabled on device " << device_id << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "SCRFD: Failed to enable CUDA Provider: " << e.what() << std::endl;
            std::cerr << "SCRFD: Falling back to CPU." << std::endl;
        }
    } else {
        std::cout << "SCRFD: CPU Mode requested." << std::endl;
    }

    input_size = input_size_value > 0 ? input_size_value : 640;
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
}

void SCRFDExtractor::GenerateAnchors(int height, int width, std::vector<std::vector<float>>& anchors) {
    // Generate anchor centers for each stride
    anchors.clear();
    for (int stride : strides) {
        int h = (height + stride - 1) / stride;
        int w = (width + stride - 1) / stride;
        
        std::vector<float> anchors_stride;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                anchors_stride.push_back((float)(x * stride));
                anchors_stride.push_back((float)(y * stride));
            }
        }
        anchors.push_back(anchors_stride);
    }
}

std::vector<Face> SCRFDExtractor::Detect(const cv::Mat& img, float threshold) {
    // Preprocess
    int h = img.rows;
    int w = img.cols;
    int max_dim = std::max(h, w);
    float scale = static_cast<float>(input_size) / max_dim;
    int new_w = (int)(w * scale);
    int new_h = (int)(h * scale);
    
    // Ensure new size is divisible by 32
    new_w = (new_w / 32) * 32;
    new_h = (new_h / 32) * 32;
    scale = (float)new_w / w; // Recalculate scale

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    
    // Convert to float and normalize
    cv::Mat input_blob;
    resized.convertTo(input_blob, CV_32FC3);
    input_blob -= cv::Scalar(127.5, 127.5, 127.5);
    input_blob /= 128.0;

    // NHWC -> NCHW
    std::vector<int64_t> input_shape = {1, 3, new_h, new_w};
    size_t input_tensor_size = 1 * 3 * new_h * new_w;
    std::vector<float> input_tensor_values(input_tensor_size);
    
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            cv::Vec3f pixel = input_blob.at<cv::Vec3f>(y, x);
            input_tensor_values[0 * new_h * new_w + y * new_w + x] = pixel[2]; // R
            input_tensor_values[1 * new_h * new_w + y * new_w + x] = pixel[1]; // G
            input_tensor_values[2 * new_h * new_w + y * new_w + x] = pixel[0]; // B
        }
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size());

    // Process outputs
    std::vector<OutputTensorInfo> outputs_data;
    for (size_t i = 0; i < output_tensors.size(); i++) {
        float* floatarr = output_tensors[i].GetTensorMutableData<float>();
        auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        outputs_data.push_back({floatarr, shape});
    }

    // Generate anchors for current size
    std::vector<std::vector<float>> anchors;
    GenerateAnchors(new_h, new_w, anchors);

    std::vector<Face> faces;
    PostProcess(outputs_data, anchors, scale, faces, threshold);
    
    return faces;
}

void SCRFDExtractor::PostProcess(const std::vector<OutputTensorInfo>& outputs, 
                                 const std::vector<std::vector<float>>& anchors,
                                 float scale, std::vector<Face>& faces, float threshold) {
    // Identify outputs by shape
    // Shape is typically [1, N, C] or [N, C]
    
    struct StrideOutputs {
        const float* score = nullptr;
        const float* bbox = nullptr;
        int num_anchors = 0;
    };
    
    // Map: Stride -> Outputs
    std::map<int, StrideOutputs> stride_map;
    
    // We expect 3 strides: 8, 16, 32
    // Stride 8: Largest N
    // Stride 16: Medium N
    // Stride 32: Smallest N
    
    // Group by N (number of anchors)
    std::map<int, std::vector<int>> by_num_anchors;
    
    for (int i = 0; i < (int)outputs.size(); i++) {
        const auto& dims = outputs[i].dims;
        // Last dim is channels?
        // Shape could be [12800, 1] -> dims[0]=12800, dims[1]=1
        // Or [1, 12800, 1] -> dims[1]=12800
        
        int channels = 0;
        int num_anchors = 0;
        
        if (dims.size() >= 2) {
            channels = (int)dims.back();
            num_anchors = (int)dims[dims.size() - 2];
        }
        
        if (num_anchors > 0) {
            by_num_anchors[num_anchors].push_back(i);
        }
    }
    
    // We expect 3 groups of N.
    if (by_num_anchors.size() < 3) return;
    
    // Sort groups by N descending -> Stride 8, 16, 32
    // Map is sorted by key (N) ascending?
    // Smallest N -> Stride 32
    // Largest N -> Stride 8
    
    auto it = by_num_anchors.rbegin(); // Largest N
    int stride_order[3] = {8, 16, 32};
    
    for (int s = 0; s < 3; s++) {
        if (it == by_num_anchors.rend()) break;
        
        int n = it->first;
        const auto& indices = it->second;
        int current_stride = stride_order[s];
        
        StrideOutputs so;
        so.num_anchors = n;
        
        for (int idx : indices) {
            int channels = (int)outputs[idx].dims.back();
            if (channels == 1) so.score = outputs[idx].data;
            else if (channels == 4) so.bbox = outputs[idx].data;
        }
        
        if (so.score && so.bbox) {
            stride_map[current_stride] = so;
        }
        
        it++;
    }
    
    for (int s : stride_order) {
        if (stride_map.find(s) == stride_map.end()) continue;
        
        const auto& so = stride_map[s];
        int num_grid_points = (int)anchors[s == 8 ? 0 : (s == 16 ? 1 : 2)].size() / 2;
        
        // Safety check
        if (num_grid_points == 0) continue;
        
        int anchors_per_point = so.num_anchors / num_grid_points;
        if (anchors_per_point <= 0) continue;
        
        const std::vector<float>& current_anchors = anchors[s == 8 ? 0 : (s == 16 ? 1 : 2)];
        
        for (int i = 0; i < num_grid_points; i++) {
            for (int k = 0; k < anchors_per_point; k++) {
                int idx = i * anchors_per_point + k;
                float score = so.score[idx];
                
                if (score < threshold) continue;
                
                float cx = current_anchors[2*i];
                float cy = current_anchors[2*i+1];
                
                int bbox_idx = idx * 4;
                float l = so.bbox[bbox_idx + 0] * s;
                float t = so.bbox[bbox_idx + 1] * s;
                float r = so.bbox[bbox_idx + 2] * s;
                float b = so.bbox[bbox_idx + 3] * s;
                
                float x1 = cx - l;
                float y1 = cy - t;
                float x2 = cx + r;
                float y2 = cy + b;
                
                faces.push_back({x1 / scale, y1 / scale, x2 / scale, y2 / scale, score});
            }
        }
    }
    
    NMS(faces, 0.4f);
}

void SCRFDExtractor::NMS(std::vector<Face>& faces, float threshold) {
    if (faces.empty()) return;
    std::sort(faces.begin(), faces.end(), [](const Face& a, const Face& b) {
        return a.score > b.score;
    });
    
    std::vector<bool> suppressed(faces.size(), false);
    std::vector<Face> result;
    
    for (size_t i = 0; i < faces.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(faces[i]);
        
        float area_i = (faces[i].x2 - faces[i].x1) * (faces[i].y2 - faces[i].y1);
        
        for (size_t j = i + 1; j < faces.size(); j++) {
            if (suppressed[j]) continue;
            
            float xx1 = std::max(faces[i].x1, faces[j].x1);
            float yy1 = std::max(faces[i].y1, faces[j].y1);
            float xx2 = std::min(faces[i].x2, faces[j].x2);
            float yy2 = std::min(faces[i].y2, faces[j].y2);
            
            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float inter = w * h;
            
            float area_j = (faces[j].x2 - faces[j].x1) * (faces[j].y2 - faces[j].y1);
            float union_area = area_i + area_j - inter;
            
            if (inter / union_area > threshold) {
                suppressed[j] = true;
            }
        }
    }
    faces = result;
}
