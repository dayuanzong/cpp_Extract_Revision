#pragma once

#ifdef _WIN32
    #ifdef FACE_EXTRACTOR_EXPORTS
        #define FACE_API __declspec(dllexport)
    #else
        #define FACE_API __declspec(dllimport)
    #endif
#else
    #define FACE_API
#endif

extern "C" {

    struct FaceInfo {
        unsigned char* jpg_data;
        int jpg_size;
        float landmarks[136];      // Source landmarks (68 * 2)
        float aligned_landmarks[136]; // Aligned landmarks (68 * 2)
        int embedding_dim;
        float embedding[512];
        int target_index;
        float target_sim;
        bool is_target;
        float source_rect[4];      // Left, Top, Right, Bottom
        float detect_score;
        float blur_variance;       // Sharpness metric
        char pose_tag[32];         // "抬头", "低头" etc.
        float pitch;
        float yaw;
        float roll;
        int blur_class;
        float mouth_value;
        bool mouth_open;
        bool valid;
    };

    // Error Codes
    const int FACE_OK = 0;
    const int FACE_ERR_MODEL_LOAD = -1;
    const int FACE_ERR_INVALID_INPUT = -2;
    const int FACE_ERR_INFERENCE = -3;
    const int FACE_ERR_UNKNOWN = -5;

    // API
    // Initialize pipeline with model directory
    FACE_API int InitPipeline(const wchar_t* model_dir, int device_id);
    FACE_API int InitPipelineEx(const wchar_t* model_dir, int device_id, const wchar_t* s3fd_path, const wchar_t* fan_path, const wchar_t* insight_path);
    
    // Release all resources
    FACE_API void ReleasePipeline();
    
    // Process image: returns array of FaceInfo via out_faces pointer
    // Python needs to call FreeFaceResults on the returned pointer
    FACE_API int ProcessImage(const wchar_t* img_path, FaceInfo** out_faces, int* out_count, int face_type = 2); // Default to FULL (2)

    FACE_API int ProcessImageMat(const unsigned char* bgr, int width, int height, int step, FaceInfo** out_faces, int* out_count, int face_type = 2);
    
    // Free the array returned by ProcessImage
    FACE_API void FreeFaceResults(FaceInfo* faces, int count);

    FACE_API int ReadImageBGR(const wchar_t* img_path, unsigned char** out_data, int* out_width, int* out_height, int* out_channels, int* out_step);
    FACE_API int DecodeImageBuffer(const unsigned char* data, int size, unsigned char** out_data, int* out_width, int* out_height, int* out_channels, int* out_step);
    FACE_API int EncodeImageBuffer(const unsigned char* data, int width, int height, int channels, int format, int quality, unsigned char** out_data, int* out_size);
    FACE_API int InsertApp15Jpeg(const unsigned char* jpg_data, int jpg_size, const unsigned char* app15_data, int app15_size, unsigned char** out_data, int* out_size);
    FACE_API int ExtractApp15Jpeg(const unsigned char* jpg_data, int jpg_size, unsigned char** out_data, int* out_size);
    FACE_API void FreeImageBuffer(unsigned char* data);

    FACE_API int SetFilterParams(int enable_blur, float blur_low, float blur_high,
                                 int enable_pose, float pitch_threshold, float yaw_threshold,
                                 int enable_mouth, float mouth_threshold);
    FACE_API int SetAlignSize(int size);
    FACE_API int SetMaxFaces(int max_faces);
    FACE_API int SetJpegQuality(int quality);

    FACE_API int EmbeddingBestMatch(const float* emb, int emb_dim,
                                    const float* refs, int ref_count,
                                    int* out_best_index, float* out_best_sim);
    FACE_API int GetEmbeddingDim(int* out_dim);
    FACE_API int ExtractEmbedding(const unsigned char* bgr, int width, int height, int step,
                                  const float* landmarks, int landmarks_len,
                                  float* out_emb, int emb_dim);
    FACE_API int SetReferenceEmbeddings(const float* refs, int ref_count, int ref_dim, float sim_threshold);
    FACE_API int ClearReferenceEmbeddings();

    FACE_API int GetVideoInfo(const wchar_t* video_path, int* out_frame_count, double* out_fps, int* out_width, int* out_height);
    FACE_API int ReadVideoFrame(const wchar_t* video_path, int frame_index, unsigned char** out_data, int* out_width, int* out_height, int* out_channels, int* out_step);

}
