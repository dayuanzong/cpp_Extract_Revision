#include "utils.h"
#include "S3FDExtractor.h"
#include "FANExtractor.h"
#include <iostream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    try {
        // Paths
        std::wstring project_root = fs::current_path().wstring();
        std::wstring model_dir = project_root + L"\\models_exported\\";
        std::wstring data_dir = project_root + L"\\models_exported\\verification_data\\";
        
        // Load Image
        // "原版提取" = \u539f\u7248\u63d0\u53d6
        std::wstring raw_img_path = project_root + L"\\data\\input\\\u539f\u7248\u63d0\u53d6\\00001_0.jpg";
        
        std::wcout << L"Loading image from: " << raw_img_path << std::endl;
        cv::Mat img = ReadImage(raw_img_path);
        if (img.empty()) {
            std::wcerr << L"Failed to load image: " << raw_img_path << std::endl;
            return 1;
        }
        std::cout << "Image loaded: " << img.cols << "x" << img.rows << std::endl;

        // Initialize S3FD
        std::wstring s3fd_path = model_dir + L"S3FD.onnx";
        std::wcout << L"Loading S3FD from " << s3fd_path << std::endl;
        S3FDExtractor s3fd(s3fd_path);
        
        // Run S3FD
        std::vector<Face> faces = s3fd.Detect(img);
        std::cout << "Detected " << faces.size() << " faces." << std::endl;
        
        if (faces.empty()) {
            std::cerr << "No faces detected!" << std::endl;
            return 1;
        }
        
        Face face = faces[0];
        std::cout << "Face 0: [" << face.x1 << ", " << face.y1 << ", " << face.x2 << ", " << face.y2 << "]" << std::endl;

        // Initialize FAN
        std::wstring fan_path = model_dir + L"2DFAN.onnx";
        std::wcout << L"Loading FAN from " << fan_path << std::endl;
        FANExtractor fan(fan_path);
        
        // Run FAN
        // We need to convert Face struct to Rect
        cv::Rect2f rect(face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1);
        std::vector<Point2f> landmarks = fan.Extract(img, rect);
        
        std::cout << "Extracted " << landmarks.size() << " landmarks." << std::endl;
        
        // Verify with Python data (optional but good)
        // For now just print first few landmarks
        for (int i = 0; i < std::min((int)landmarks.size(), 5); ++i) {
            std::cout << "Landmark " << i << ": (" << landmarks[i].x << ", " << landmarks[i].y << ")" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
