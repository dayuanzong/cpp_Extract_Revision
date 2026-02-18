#include "FANExtractor.h"
#include <cmath>

FANExtractor::FANExtractor(const std::wstring& model_path, int device_id)
    : env(ORT_LOGGING_LEVEL_WARNING, "FAN") {

    Ort::SessionOptions session_options;

    if (device_id >= 0) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = device_id;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "FAN: CUDA Provider enabled on device " << device_id << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "FAN: Failed to enable CUDA Provider: " << e.what() << std::endl;
            std::cerr << "FAN: Falling back to CPU." << std::endl;
        }
    } else {
        std::cout << "FAN: CPU Mode requested." << std::endl;
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
    }
}

static cv::Point2f TransformPoint(const cv::Point2f& point, const cv::Point2f& center, float scale, float resolution) {
    float h = 200.0f * scale;
    cv::Matx33f m = cv::Matx33f::eye();
    m(0, 0) = resolution / h;
    m(1, 1) = resolution / h;
    m(0, 2) = resolution * (-center.x / h + 0.5f);
    m(1, 2) = resolution * (-center.y / h + 0.5f);
    cv::Matx33f inv = m.inv();
    cv::Vec3f pt(point.x, point.y, 1.0f);
    cv::Vec3f res = inv * pt;
    return cv::Point2f(res[0], res[1]);
}

std::vector<Point2f> FANExtractor::Extract(const cv::Mat& img, const cv::Rect2f& face_rect, bool multi_sample) {
    float scale = (face_rect.width + face_rect.height) / 195.0f;
    cv::Point2f center(face_rect.x + face_rect.width / 2.0f, face_rect.y + face_rect.height / 2.0f);

    std::vector<cv::Point2f> centers;
    centers.push_back(center);
    if (multi_sample) {
        centers.push_back(center + cv::Point2f(-1.0f, -1.0f));
        centers.push_back(center + cv::Point2f(1.0f, -1.0f));
        centers.push_back(center + cv::Point2f(1.0f, 1.0f));
        centers.push_back(center + cv::Point2f(-1.0f, 1.0f));
    }

    std::vector<Point2f> accum(68, cv::Point2f(0.0f, 0.0f));
    int count = 0;
    for (const auto& c : centers) {
        auto pts = ExtractSingle(img, c, scale);
        if (pts.size() != 68) continue;
        for (int i = 0; i < 68; i++) {
            accum[i].x += pts[i].x;
            accum[i].y += pts[i].y;
        }
        count++;
    }

    if (count == 0) return {};
    for (int i = 0; i < 68; i++) {
        accum[i].x /= (float)count;
        accum[i].y /= (float)count;
    }
    return accum;
}

std::vector<Point2f> FANExtractor::ExtractSingle(const cv::Mat& img, cv::Point2f center, float scale) {
    cv::Mat cropped_img = Crop(img, center, scale);

    cv::Mat input_blob;
    cropped_img.convertTo(input_blob, CV_32FC3, 1.0f / 255.0f);

    std::vector<int64_t> input_shape = {1, 256, 256, 3};
    size_t input_tensor_size = 1 * 256 * 256 * 3;
    std::vector<float> input_tensor_values(input_tensor_size);

    for (int y = 0; y < 256; y++) {
        for (int x = 0; x < 256; x++) {
            cv::Vec3f pixel = input_blob.at<cv::Vec3f>(y, x);
            input_tensor_values[(y * 256 + x) * 3 + 0] = pixel[2];
            input_tensor_values[(y * 256 + x) * 3 + 1] = pixel[1];
            input_tensor_values[(y * 256 + x) * 3 + 2] = pixel[0];
        }
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size());

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    std::vector<float> heatmap(output_data, output_data + 68 * 64 * 64);

    return PostProcess(heatmap, center, scale);
}

cv::Mat FANExtractor::Crop(const cv::Mat& img, cv::Point2f center, float scale) {
    float resolution = 256.0f;

    cv::Point2f ul = TransformPoint(cv::Point2f(1.0f, 1.0f), center, scale, resolution);
    cv::Point2f br = TransformPoint(cv::Point2f(resolution, resolution), center, scale, resolution);

    int ul_x = (int)ul.x;
    int ul_y = (int)ul.y;
    int br_x = (int)br.x;
    int br_y = (int)br.y;

    int new_w = br_x - ul_x;
    int new_h = br_y - ul_y;
    if (new_w <= 0 || new_h <= 0) {
        return cv::Mat((int)resolution, (int)resolution, img.type(), cv::Scalar::all(0));
    }

    cv::Mat newImg = cv::Mat::zeros(new_h, new_w, img.type());

    int ht = img.rows;
    int wd = img.cols;

    int new_x0 = std::max(1, -ul_x + 1);
    int new_x1 = std::min(br_x, wd) - ul_x;
    int new_y0 = std::max(1, -ul_y + 1);
    int new_y1 = std::min(br_y, ht) - ul_y;

    int old_x0 = std::max(1, ul_x + 1);
    int old_x1 = std::min(br_x, wd);
    int old_y0 = std::max(1, ul_y + 1);
    int old_y1 = std::min(br_y, ht);

    if (new_x1 > new_x0 && new_y1 > new_y0 && old_x1 > old_x0 && old_y1 > old_y0) {
        cv::Rect new_roi(new_x0 - 1, new_y0 - 1, new_x1 - new_x0, new_y1 - new_y0);
        cv::Rect old_roi(old_x0 - 1, old_y0 - 1, old_x1 - old_x0, old_y1 - old_y0);
        img(old_roi).copyTo(newImg(new_roi));
    }

    cv::Mat cropped;
    cv::resize(newImg, cropped, cv::Size((int)resolution, (int)resolution), 0, 0, cv::INTER_LINEAR);
    return cropped;
}

std::vector<Point2f> FANExtractor::PostProcess(const std::vector<float>& heatmap, cv::Point2f center, float scale) {
    std::vector<Point2f> landmarks;
    int res = 64;
    
    for (int i = 0; i < 68; i++) {
        // Find max in heatmap[i]
        int max_idx = 0;
        float max_val = -1e9;
        
        for (int j = 0; j < res * res; j++) {
            float val = heatmap[i * res * res + j];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        
        int y = max_idx / res;
        int x = max_idx % res;
        
        // Refine
        float off_x = 0;
        float off_y = 0;
        
        if (x > 0 && x < res - 1 && y > 0 && y < res - 1) {
            float hm_l = heatmap[i * res * res + y * res + (x - 1)];
            float hm_r = heatmap[i * res * res + y * res + (x + 1)];
            float hm_u = heatmap[i * res * res + (y - 1) * res + x];
            float hm_d = heatmap[i * res * res + (y + 1) * res + x];
            
            float diff_x = hm_r - hm_l;
            float diff_y = hm_d - hm_u;
            
            if (diff_x > 0) off_x = 0.25;
            if (diff_x < 0) off_x = -0.25;
            if (diff_y > 0) off_y = 0.25;
            if (diff_y < 0) off_y = -0.25;
        }
        
        float px = x + 0.5f + off_x;
        float py = y + 0.5f + off_y;

        cv::Point2f orig = TransformPoint(cv::Point2f(px, py), center, scale, (float)res);
        landmarks.push_back(orig);
    }
    
    return landmarks;
}
