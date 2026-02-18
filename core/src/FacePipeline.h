#pragma once
#include "interface.h"
#include "S3FDExtractor.h"
#include "SCRFDExtractor.h"
#include "FANExtractor.h"
#include "InsightFaceLandmark.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class InsightFaceExtractor;

class FacePipeline {
public:
    FacePipeline(const std::wstring& model_dir, int device_id);
    FacePipeline(const std::wstring& model_dir, int device_id, const std::wstring& s3fd_path, const std::wstring& fan_path, const std::wstring& insight_path);
    ~FacePipeline();

    std::vector<FaceInfo> Process(const std::wstring& img_path, int face_type = 2);
    std::vector<FaceInfo> ProcessMat(const cv::Mat& img, int face_type = 2);
    void FreeFaceInfo(FaceInfo& info);
    void SetFilterParams(bool enable_blur, float blur_low, float blur_high,
                         bool enable_pose, float pitch_threshold, float yaw_threshold,
                         bool enable_mouth, float mouth_threshold);
    void SetAlignSize(int size);
    void SetMaxFaces(int max_faces);
    void SetJpegQuality(int quality);
    int GetEmbeddingDim() const;
    bool ExtractEmbedding(const cv::Mat& img, const std::vector<cv::Point2f>& landmarks, std::vector<float>& out);
    void SetReferenceEmbeddings(const float* refs, int ref_count, int ref_dim, float sim_threshold);
    void ClearReferenceEmbeddings();

private:
    S3FDExtractor* s3fd = nullptr;
    SCRFDExtractor* scrfd = nullptr;
    FANExtractor* fan = nullptr;
    InsightFaceLandmark* insight_landmark = nullptr;
    InsightFaceExtractor* insight = nullptr;
    bool insight_landmark_is_3d = false;

    cv::Mat AlignFace(const cv::Mat& img, const std::vector<cv::Point2f>& landmarks, int size, int face_type);
    float ComputeBlur(const cv::Mat& img, const std::vector<cv::Point2f>& landmarks);
    std::string EstimatePose(const std::vector<cv::Point2f>& landmarks, float& pitch, float& yaw, float& roll);
    bool IsMouthOpen(const std::vector<cv::Point2f>& landmarks, float threshold);
    
    // Helper for Umeyama
    cv::Mat GetTransformMat(const std::vector<cv::Point2f>& src_points, int size, int face_type);

    bool enable_blur = false;
    bool enable_pose = false;
    bool enable_mouth = false;
    float blur_low = 10.0f;
    float blur_high = 20.0f;
    float pitch_threshold = 15.0f;
    float yaw_threshold = 15.0f;
    float mouth_threshold = 15.0f;
    int align_size = 256;
    int max_faces = 0;
    int jpeg_quality = 90;
    std::vector<float> reference_embeddings;
    int reference_count = 0;
    int reference_dim = 0;
    float reference_threshold = 0.4f;
};
