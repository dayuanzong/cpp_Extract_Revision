#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

// Helper to read images with Unicode paths (Windows)
inline cv::Mat ReadImage(const std::wstring& path) {
    // Open file stream
    FILE* f = _wfopen(path.c_str(), L"rb");
    if (!f) {
        std::wcerr << L"Failed to open file: " << path << std::endl;
        return cv::Mat();
    }
    
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    std::vector<uchar> buf(sz);
    fread(buf.data(), 1, sz, f);
    fclose(f);
    
    return cv::imdecode(buf, cv::IMREAD_COLOR);
}

// Helper to save npy (simple version for verification data)
// We will use a library or just save debug text for now, 
// or use CNPY if we want to be fancy. 
// For this task, we just need to load reference data to compare, 
// or save our output to compare with python.
// Let's implement a simple .npy loader if needed, or just print results.

// Function to calculate IO U/Distance for verification
inline float L2Distance(const cv::Mat& a, const cv::Mat& b) {
    return cv::norm(a, b, cv::NORM_L2);
}
