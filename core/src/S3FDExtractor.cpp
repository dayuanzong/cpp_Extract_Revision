#include "S3FDExtractor.h"
#include <cmath>
#include <iostream>

S3FDExtractor::S3FDExtractor(const std::wstring& model_path, int device_id)
    : env(ORT_LOGGING_LEVEL_WARNING, "S3FD") {
    
    Ort::SessionOptions session_options;
    
    if (device_id >= 0) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = device_id;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "S3FD: CUDA Provider enabled on device " << device_id << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "S3FD: Failed to enable CUDA Provider: " << e.what() << std::endl;
            std::cerr << "S3FD: Falling back to CPU." << std::endl;
        }
    } else {
        std::cout << "S3FD: CPU Mode requested." << std::endl;
    }

    session = Ort::Session(env, model_path.c_str(), session_options);
    
    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
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
        // std::cout << "S3FD Output " << i << ": " << output_names_str.back() << std::endl;
    }
}

std::vector<Face> S3FDExtractor::Detect(const cv::Mat& img, float threshold) {
    // Preprocessing
    
    int h = img.rows;
    int w = img.cols;
    int d = std::max(w, h);
    float scale_to;
    
    if (d >= 1280) scale_to = 640.0f;
    else scale_to = d / 2.0f;
    scale_to = std::max(64.0f, scale_to);
    
    float input_scale = d / scale_to;
    int new_w = (int)(w / input_scale);
    int new_h = (int)(h / input_scale);
    
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // std::cout << "Input Scale: " << input_scale << ", New Size: " << new_w << "x" << new_h << std::endl;
    
    // Prepare input tensor
    // Batch size 1, H, W, C
    std::vector<int64_t> input_shape = {1, new_h, new_w, 3};
    size_t input_tensor_size = 1 * new_h * new_w * 3;
    std::vector<float> input_tensor_values(input_tensor_size);
    
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            cv::Vec3b pixel = resized_img.at<cv::Vec3b>(y, x);
            float r = (float)pixel[2]; // pixel[2] is R
            float g = (float)pixel[1]; // pixel[1] is G
            float b = (float)pixel[0]; // pixel[0] is B
            
            input_tensor_values[(y * new_w + x) * 3 + 0] = r;
            input_tensor_values[(y * new_w + x) * 3 + 1] = g;
            input_tensor_values[(y * new_w + x) * 3 + 2] = b;
        }
    }
    
    // Create tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());
    
    // Run
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size());
    
    // Process outputs
    std::vector<std::vector<float>> outputs_data;
    for (size_t i = 0; i < output_tensors.size(); i++) {
        float* floatarr = output_tensors[i].GetTensorMutableData<float>();
        size_t count = output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
        outputs_data.emplace_back(floatarr, floatarr + count);
    }
    
    std::vector<Face> faces;
    
    for (size_t i = 0; i < output_tensors.size() / 2; i++) {
        auto cls_info = output_tensors[2*i].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> cls_shape = cls_info.GetShape();
        
        int H = (int)cls_shape[1];
        int W = (int)cls_shape[2];
        int C = (int)cls_shape[3];

        // std::cout << "Output " << 2*i << " Shape: [" << cls_shape[0] << ", " << H << ", " << W << ", " << C << "]" << std::endl;
        
        int stride = 1 << (int)(i + 2);
        
        float* cls_data = outputs_data[2*i].data();
        float* reg_data = outputs_data[2*i+1].data();
        
        // Iterate over H, W
        for (int h_idx = 0; h_idx < H; h_idx++) {
            for (int w_idx = 0; w_idx < W; w_idx++) {
                // Check score (class 1)
                int idx = (h_idx * W + w_idx) * C + 1;
                float score = cls_data[idx];
                
                if (score > 0.05f) {
                    int reg_idx = (h_idx * W + w_idx) * 4;
                    float* loc = &reg_data[reg_idx];
                    
                    float s_d2 = stride / 2.0f;
                    float s_m4 = stride * 4.0f;
                    float priors[4] = { (float)w_idx * stride + s_d2, (float)h_idx * stride + s_d2, s_m4, s_m4 };
                    
                    float box[4];
                    box[0] = priors[0] + loc[0] * 0.1f * priors[2]; // cx
                    box[1] = priors[1] + loc[1] * 0.1f * priors[3]; // cy
                    box[2] = priors[2] * exp(loc[2] * 0.2f); // w
                    box[3] = priors[3] * exp(loc[3] * 0.2f); // h
                    
                    if (score > 0.6f) {
                         // std::cout << "Candidate: Layer " << i << " Stride " << stride << " Score " << score 
                         //          << " Box [" << box[0] << "," << box[1] << "," << box[2] << "," << box[3] << "]" 
                         //          << " w_idx " << w_idx << " h_idx " << h_idx << std::endl;
                    }

                    float x1 = box[0] - box[2] / 2.0f;
                    float y1 = box[1] - box[3] / 2.0f;
                    float x2 = box[0] + box[2] / 2.0f; // Python: box[2:] += box[:2] -> w + x1 = x2
                    float y2 = box[1] + box[3] / 2.0f;
                    
                    faces.push_back({x1, y1, x2, y2, score});
                }
            }
        }
    }
    
    NMS(faces, 0.3f);
    
    std::vector<Face> final_faces;
    for (const auto& face : faces) {
        if (face.score >= 0.5f) {
            Face f = face;
            
            int x1_int = (int)f.x1;
            int y1_int = (int)f.y1;
            int x2_int = (int)f.x2;
            int y2_int = (int)f.y2;

            f.x1 = (float)((int)(x1_int * input_scale));
            f.y1 = (float)((int)(y1_int * input_scale));
            f.x2 = (float)((int)(x2_int * input_scale));
            f.y2 = (float)((int)(y2_int * input_scale));
            
            float w_f = f.x2 - f.x1;
            float h_f = f.y2 - f.y1;
            
            if (std::min(w_f, h_f) < 40.0f) continue;
            
            f.y2 += h_f * 0.1f;
            
            final_faces.push_back(f);
        }
    }
    
    // Sort by area (descending) - Python logic
    std::sort(final_faces.begin(), final_faces.end(), [](const Face& a, const Face& b) {
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        return area_a > area_b;
    });

    return final_faces;
}

void S3FDExtractor::NMS(std::vector<Face>& faces, float threshold) {
    if (faces.empty()) return;
    
    std::sort(faces.begin(), faces.end(), [](const Face& a, const Face& b) {
        return a.score > b.score;
    });
    
    std::vector<Face> keep;
    std::vector<bool> suppressed(faces.size(), false);
    
    for (size_t i = 0; i < faces.size(); i++) {
        if (suppressed[i]) continue;
        keep.push_back(faces[i]);
        
        float area_i = (faces[i].x2 - faces[i].x1 + 1) * (faces[i].y2 - faces[i].y1 + 1);
        
        for (size_t j = i + 1; j < faces.size(); j++) {
            if (suppressed[j]) continue;
            
            float xx1 = std::max(faces[i].x1, faces[j].x1);
            float yy1 = std::max(faces[i].y1, faces[j].y1);
            float xx2 = std::min(faces[i].x2, faces[j].x2);
            float yy2 = std::min(faces[i].y2, faces[j].y2);
            
            float w = std::max(0.0f, xx2 - xx1 + 1);
            float h = std::max(0.0f, yy2 - yy1 + 1);
            
            float inter = w * h;
            float area_j = (faces[j].x2 - faces[j].x1 + 1) * (faces[j].y2 - faces[j].y1 + 1);
            float ovr = inter / (area_i + area_j - inter);
            
            if (ovr > threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    faces = keep;
}

void S3FDExtractor::PostProcess(const std::vector<std::vector<float>>& outputs, 
                               int width, int height, float scale, 
                               std::vector<Face>& faces, float threshold) {
    // Moved logic to Detect for easier access to shapes
}
