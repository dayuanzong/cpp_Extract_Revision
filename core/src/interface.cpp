#include "interface.h"
#include "FacePipeline.h"
#include <memory>
#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <cmath>
#ifdef _WIN32
#include <windows.h>
#endif

std::unique_ptr<FacePipeline> g_pipeline;

static std::string WideToUtf8(const std::wstring& wstr) {
#ifdef _WIN32
    if (wstr.empty()) return std::string();
    int size = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), nullptr, 0, nullptr, nullptr);
    if (size <= 0) return std::string();
    std::string result(size, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), &result[0], size, nullptr, nullptr);
    return result;
#else
    return std::string(wstr.begin(), wstr.end());
#endif
}

static bool OpenVideoCapture(const std::string& path, cv::VideoCapture& cap) {
    if (cap.open(path, cv::CAP_FFMPEG)) return true;
#ifdef _WIN32
    if (cap.open(path, cv::CAP_MSMF)) return true;
    if (cap.open(path, cv::CAP_DSHOW)) return true;
#endif
    return false;
}

static int FillResults(const std::vector<FaceInfo>& results, FaceInfo** out_faces, int* out_count) {
    *out_count = (int)results.size();
    if (*out_count == 0) {
        *out_faces = nullptr;
        return FACE_OK;
    }
    *out_faces = new FaceInfo[*out_count];
    for (int i = 0; i < *out_count; i++) {
        (*out_faces)[i] = results[i];
    }
    return FACE_OK;
}

static int CopyMatToBuffer(const cv::Mat& img_in, unsigned char** out_data, int* out_width, int* out_height, int* out_channels, int* out_step) {
    if (!out_data || !out_width || !out_height || !out_channels || !out_step) return FACE_ERR_INVALID_INPUT;
    if (img_in.empty()) return FACE_ERR_INFERENCE;

    cv::Mat img = img_in;
    if (img.depth() != CV_8U) {
        cv::Mat converted;
        img.convertTo(converted, CV_8U);
        img = converted;
    }

    int channels = img.channels();
    if (channels != 1 && channels != 3 && channels != 4) return FACE_ERR_INVALID_INPUT;

    size_t size = img.step * img.rows;
    if (size == 0) return FACE_ERR_INFERENCE;

    unsigned char* buf = new unsigned char[size];
    memcpy(buf, img.data, size);

    *out_data = buf;
    *out_width = img.cols;
    *out_height = img.rows;
    *out_channels = channels;
    *out_step = (int)img.step;

    return FACE_OK;
}

FACE_API int InitPipeline(const wchar_t* model_dir, int device_id) {
    try {
        if (!model_dir) return FACE_ERR_INVALID_INPUT;
        
        g_pipeline = std::make_unique<FacePipeline>(model_dir, device_id);
        return FACE_OK;
    } catch (const std::exception& e) {
        std::cerr << "InitPipeline Error: " << e.what() << std::endl;
        return FACE_ERR_MODEL_LOAD;
    } catch (...) {
        return FACE_ERR_UNKNOWN;
    }
}

FACE_API int InitPipelineEx(const wchar_t* model_dir, int device_id, const wchar_t* s3fd_path, const wchar_t* fan_path, const wchar_t* insight_path) {
    try {
        if (!model_dir) return FACE_ERR_INVALID_INPUT;
        std::wstring s3fd = s3fd_path ? s3fd_path : L"";
        std::wstring fan = fan_path ? fan_path : L"";
        std::wstring insight = insight_path ? insight_path : L"";
        g_pipeline = std::make_unique<FacePipeline>(model_dir, device_id, s3fd, fan, insight);
        return FACE_OK;
    } catch (const std::exception& e) {
        std::cerr << "InitPipelineEx Error: " << e.what() << std::endl;
        return FACE_ERR_MODEL_LOAD;
    } catch (...) {
        return FACE_ERR_UNKNOWN;
    }
}

FACE_API void ReleasePipeline() {
    g_pipeline.reset();
}

FACE_API int SetFilterParams(int enable_blur, float blur_low, float blur_high,
                             int enable_pose, float pitch_threshold, float yaw_threshold,
                             int enable_mouth, float mouth_threshold) {
    if (!g_pipeline) return FACE_ERR_MODEL_LOAD;
    g_pipeline->SetFilterParams(enable_blur != 0, blur_low, blur_high,
                                enable_pose != 0, pitch_threshold, yaw_threshold,
                                enable_mouth != 0, mouth_threshold);
    return FACE_OK;
}

FACE_API int SetAlignSize(int size) {
    if (!g_pipeline) return FACE_ERR_MODEL_LOAD;
    g_pipeline->SetAlignSize(size);
    return FACE_OK;
}

FACE_API int SetMaxFaces(int max_faces) {
    if (!g_pipeline) return FACE_ERR_MODEL_LOAD;
    g_pipeline->SetMaxFaces(max_faces);
    return FACE_OK;
}

FACE_API int SetJpegQuality(int quality) {
    if (!g_pipeline) return FACE_ERR_MODEL_LOAD;
    g_pipeline->SetJpegQuality(quality);
    return FACE_OK;
}

FACE_API int EmbeddingBestMatch(const float* emb, int emb_dim,
                                const float* refs, int ref_count,
                                int* out_best_index, float* out_best_sim) {
    if (!emb || !refs || emb_dim <= 0 || ref_count <= 0 || !out_best_index || !out_best_sim) return FACE_ERR_INVALID_INPUT;

    float emb_norm = 0.0f;
    for (int i = 0; i < emb_dim; i++) {
        emb_norm += emb[i] * emb[i];
    }
    emb_norm = std::sqrt(emb_norm);
    if (emb_norm <= 0.0f) {
        *out_best_index = -1;
        *out_best_sim = -1.0f;
        return FACE_ERR_INFERENCE;
    }

    float best_sim = -1.0f;
    int best_idx = -1;

    for (int r = 0; r < ref_count; r++) {
        const float* ref = refs + r * emb_dim;
        float dot = 0.0f;
        float ref_norm = 0.0f;
        for (int i = 0; i < emb_dim; i++) {
            dot += emb[i] * ref[i];
            ref_norm += ref[i] * ref[i];
        }
        ref_norm = std::sqrt(ref_norm);
        if (ref_norm <= 0.0f) continue;
        float sim = dot / (emb_norm * ref_norm);
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = r;
        }
    }

    *out_best_index = best_idx;
    *out_best_sim = best_sim;
    return FACE_OK;
}

FACE_API int GetEmbeddingDim(int* out_dim) {
    if (!g_pipeline || !out_dim) return FACE_ERR_INVALID_INPUT;
    int dim = g_pipeline->GetEmbeddingDim();
    if (dim <= 0) return FACE_ERR_MODEL_LOAD;
    *out_dim = dim;
    return FACE_OK;
}

FACE_API int ExtractEmbedding(const unsigned char* bgr, int width, int height, int step,
                              const float* landmarks, int landmarks_len,
                              float* out_emb, int emb_dim) {
    if (!g_pipeline) return FACE_ERR_MODEL_LOAD;
    if (!bgr || width <= 0 || height <= 0 || step <= 0) return FACE_ERR_INVALID_INPUT;
    if (!landmarks || landmarks_len < 136 || !out_emb || emb_dim <= 0) return FACE_ERR_INVALID_INPUT;

    try {
        cv::Mat img(height, width, CV_8UC3, const_cast<unsigned char*>(bgr), step);
        cv::Mat img_copy = img.clone();
        std::vector<cv::Point2f> lms;
        lms.reserve(68);
        for (int i = 0; i < 68; i++) {
            lms.emplace_back(landmarks[i * 2], landmarks[i * 2 + 1]);
        }
        std::vector<float> emb;
        if (!g_pipeline->ExtractEmbedding(img_copy, lms, emb)) return FACE_ERR_INFERENCE;
        if ((int)emb.size() != emb_dim) return FACE_ERR_INVALID_INPUT;
        std::memcpy(out_emb, emb.data(), sizeof(float) * emb_dim);
        return FACE_OK;
    } catch (const std::exception& e) {
        std::cerr << "ExtractEmbedding Error: " << e.what() << std::endl;
        return FACE_ERR_INFERENCE;
    } catch (...) {
        return FACE_ERR_UNKNOWN;
    }
}

FACE_API int SetReferenceEmbeddings(const float* refs, int ref_count, int ref_dim, float sim_threshold) {
    if (!g_pipeline) return FACE_ERR_MODEL_LOAD;
    g_pipeline->SetReferenceEmbeddings(refs, ref_count, ref_dim, sim_threshold);
    return FACE_OK;
}

FACE_API int ClearReferenceEmbeddings() {
    if (!g_pipeline) return FACE_ERR_MODEL_LOAD;
    g_pipeline->ClearReferenceEmbeddings();
    return FACE_OK;
}

FACE_API int ProcessImage(const wchar_t* img_path, FaceInfo** out_faces, int* out_count, int face_type) {
    if (!g_pipeline) return FACE_ERR_MODEL_LOAD;
    if (!img_path || !out_faces || !out_count) return FACE_ERR_INVALID_INPUT;

    try {
        std::vector<FaceInfo> results = g_pipeline->Process(img_path, face_type);
        return FillResults(results, out_faces, out_count);
    } catch (const std::exception& e) {
        std::cerr << "ProcessImage Error: " << e.what() << std::endl;
        return FACE_ERR_INFERENCE;
    } catch (...) {
        return FACE_ERR_UNKNOWN;
    }
}

FACE_API int ProcessImageMat(const unsigned char* bgr, int width, int height, int step, FaceInfo** out_faces, int* out_count, int face_type) {
    if (!g_pipeline) return FACE_ERR_MODEL_LOAD;
    if (!bgr || !out_faces || !out_count || width <= 0 || height <= 0 || step <= 0) return FACE_ERR_INVALID_INPUT;

    try {
        cv::Mat img(height, width, CV_8UC3, const_cast<unsigned char*>(bgr), step);
        std::vector<FaceInfo> results = g_pipeline->ProcessMat(img, face_type);
        return FillResults(results, out_faces, out_count);
    } catch (const std::exception& e) {
        std::cerr << "ProcessImageMat Error: " << e.what() << std::endl;
        return FACE_ERR_INFERENCE;
    } catch (...) {
        return FACE_ERR_UNKNOWN;
    }
}

FACE_API void FreeFaceResults(FaceInfo* faces, int count) {
    if (!faces) return;
    
    for (int i = 0; i < count; i++) {
        if (faces[i].jpg_data) {
            delete[] faces[i].jpg_data;
            faces[i].jpg_data = nullptr;
        }
    }
    delete[] faces;
}

FACE_API int ReadImageBGR(const wchar_t* img_path, unsigned char** out_data, int* out_width, int* out_height, int* out_channels, int* out_step) {
    if (!img_path || !out_data || !out_width || !out_height || !out_channels || !out_step) return FACE_ERR_INVALID_INPUT;

    cv::Mat img;
#ifdef _WIN32
    FILE* f = _wfopen(img_path, L"rb");
    if (!f) return FACE_ERR_INVALID_INPUT;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(f);
        return FACE_ERR_INFERENCE;
    }
    std::vector<uchar> buf((size_t)sz);
    fread(buf.data(), 1, (size_t)sz, f);
    fclose(f);
    img = cv::imdecode(buf, cv::IMREAD_COLOR);
#else
    std::wstring path_ws(img_path);
    std::string path_utf8(path_ws.begin(), path_ws.end());
    img = cv::imread(path_utf8, cv::IMREAD_COLOR);
#endif
    if (img.empty()) return FACE_ERR_INFERENCE;
    return CopyMatToBuffer(img, out_data, out_width, out_height, out_channels, out_step);
}

FACE_API int GetVideoInfo(const wchar_t* video_path, int* out_frame_count, double* out_fps, int* out_width, int* out_height) {
    if (!video_path || !out_frame_count || !out_fps || !out_width || !out_height) return FACE_ERR_INVALID_INPUT;
    std::wstring path_ws(video_path);
    std::string path_utf8 = WideToUtf8(path_ws);
    if (path_utf8.empty()) return FACE_ERR_INVALID_INPUT;

    cv::VideoCapture cap;
    if (!OpenVideoCapture(path_utf8, cap)) return FACE_ERR_INFERENCE;

    *out_frame_count = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    *out_fps = cap.get(cv::CAP_PROP_FPS);
    *out_width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    *out_height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    return FACE_OK;
}

FACE_API int ReadVideoFrame(const wchar_t* video_path, int frame_index, unsigned char** out_data, int* out_width, int* out_height, int* out_channels, int* out_step) {
    if (!video_path || !out_data || !out_width || !out_height || !out_channels || !out_step) return FACE_ERR_INVALID_INPUT;
    if (frame_index < 0) return FACE_ERR_INVALID_INPUT;
    std::wstring path_ws(video_path);
    std::string path_utf8 = WideToUtf8(path_ws);
    if (path_utf8.empty()) return FACE_ERR_INVALID_INPUT;

    cv::VideoCapture cap;
    if (!OpenVideoCapture(path_utf8, cap)) return FACE_ERR_INFERENCE;
    cap.set(cv::CAP_PROP_POS_FRAMES, (double)frame_index);

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) return FACE_ERR_INFERENCE;
    return CopyMatToBuffer(frame, out_data, out_width, out_height, out_channels, out_step);
}

FACE_API int DecodeImageBuffer(const unsigned char* data, int size, unsigned char** out_data, int* out_width, int* out_height, int* out_channels, int* out_step) {
    if (!data || size <= 0) return FACE_ERR_INVALID_INPUT;
    std::vector<uchar> buf(data, data + size);
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_UNCHANGED);
    if (img.empty()) return FACE_ERR_INFERENCE;
    return CopyMatToBuffer(img, out_data, out_width, out_height, out_channels, out_step);
}

FACE_API int EncodeImageBuffer(const unsigned char* data, int width, int height, int channels, int format, int quality, unsigned char** out_data, int* out_size) {
    if (!data || !out_data || !out_size) return FACE_ERR_INVALID_INPUT;
    if (width <= 0 || height <= 0) return FACE_ERR_INVALID_INPUT;
    if (channels != 1 && channels != 3 && channels != 4) return FACE_ERR_INVALID_INPUT;

    int type = channels == 1 ? CV_8UC1 : (channels == 3 ? CV_8UC3 : CV_8UC4);
    cv::Mat img(height, width, type, const_cast<unsigned char*>(data));

    std::vector<uchar> buf;
    std::vector<int> params;

    if (format == 0) {
        if (!cv::imencode(".png", img, buf, params)) return FACE_ERR_INFERENCE;
    } else if (format == 1) {
        int q = quality;
        if (q < 0) q = 0;
        if (q > 100) q = 100;
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(q);
        if (!cv::imencode(".jpg", img, buf, params)) return FACE_ERR_INFERENCE;
    } else {
        return FACE_ERR_INVALID_INPUT;
    }

    if (buf.empty()) return FACE_ERR_INFERENCE;
    *out_size = (int)buf.size();
    unsigned char* out_buf = new unsigned char[*out_size];
    memcpy(out_buf, buf.data(), (size_t)*out_size);
    *out_data = out_buf;

    return FACE_OK;
}

FACE_API int InsertApp15Jpeg(const unsigned char* jpg_data, int jpg_size, const unsigned char* app15_data, int app15_size, unsigned char** out_data, int* out_size) {
    if (!jpg_data || jpg_size <= 4 || !app15_data || app15_size < 0 || !out_data || !out_size) return FACE_ERR_INVALID_INPUT;
    if (jpg_data[0] != 0xFF || jpg_data[1] != 0xD8) return FACE_ERR_INVALID_INPUT;
    if (app15_size + 2 > 0xFFFF) return FACE_ERR_INVALID_INPUT;

    size_t i = 2;
    size_t last_app_end = 2;

    auto read_seg_len = [&](size_t pos) -> int {
        return (jpg_data[pos] << 8) | jpg_data[pos + 1];
    };

    while (i + 1 < (size_t)jpg_size) {
        if (jpg_data[i] != 0xFF) {
            i++;
            continue;
        }

        unsigned char marker = jpg_data[i + 1];

        if (marker == 0xDA || marker == 0xD9) {
            break;
        }

        if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            i += 2;
            continue;
        }

        if (i + 3 >= (size_t)jpg_size) return FACE_ERR_INFERENCE;
        int seg_len = read_seg_len(i + 2);
        if (seg_len < 2) return FACE_ERR_INFERENCE;
        size_t seg_end = i + 2 + (size_t)seg_len;
        if (seg_end > (size_t)jpg_size) return FACE_ERR_INFERENCE;

        if (marker >= 0xE0 && marker <= 0xEF) {
            last_app_end = seg_end;
        }

        i = seg_end;
    }

    size_t insert_pos = last_app_end;
    size_t new_size = (size_t)jpg_size + (size_t)app15_size + 4;
    std::vector<unsigned char> out;
    out.reserve(new_size);

    out.insert(out.end(), jpg_data, jpg_data + insert_pos);
    out.push_back(0xFF);
    out.push_back(0xEF);
    int app_len = app15_size + 2;
    out.push_back((unsigned char)((app_len >> 8) & 0xFF));
    out.push_back((unsigned char)(app_len & 0xFF));
    out.insert(out.end(), app15_data, app15_data + app15_size);
    out.insert(out.end(), jpg_data + insert_pos, jpg_data + jpg_size);

    *out_size = (int)out.size();
    unsigned char* out_buf = new unsigned char[*out_size];
    memcpy(out_buf, out.data(), (size_t)*out_size);
    *out_data = out_buf;

    return FACE_OK;
}

FACE_API int ExtractApp15Jpeg(const unsigned char* jpg_data, int jpg_size, unsigned char** out_data, int* out_size) {
    if (!jpg_data || jpg_size <= 4 || !out_data || !out_size) return FACE_ERR_INVALID_INPUT;
    if (jpg_data[0] != 0xFF || jpg_data[1] != 0xD8) return FACE_ERR_INVALID_INPUT;

    size_t i = 2;
    auto read_seg_len = [&](size_t pos) -> int {
        return (jpg_data[pos] << 8) | jpg_data[pos + 1];
    };

    while (i + 1 < (size_t)jpg_size) {
        if (jpg_data[i] != 0xFF) {
            i++;
            continue;
        }

        unsigned char marker = jpg_data[i + 1];

        if (marker == 0xDA || marker == 0xD9) {
            break;
        }

        if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
            i += 2;
            continue;
        }

        if (i + 3 >= (size_t)jpg_size) return FACE_ERR_INFERENCE;
        int seg_len = read_seg_len(i + 2);
        if (seg_len < 2) return FACE_ERR_INFERENCE;
        size_t seg_end = i + 2 + (size_t)seg_len;
        if (seg_end > (size_t)jpg_size) return FACE_ERR_INFERENCE;

        if (marker == 0xEF) {
            size_t data_pos = i + 4;
            size_t data_size = (size_t)seg_len - 2;
            if (data_pos + data_size > (size_t)jpg_size) return FACE_ERR_INFERENCE;
            unsigned char* out_buf = new unsigned char[data_size];
            memcpy(out_buf, jpg_data + data_pos, data_size);
            *out_data = out_buf;
            *out_size = (int)data_size;
            return FACE_OK;
        }

        i = seg_end;
    }

    return FACE_ERR_INFERENCE;
}

FACE_API void FreeImageBuffer(unsigned char* data) {
    if (data) {
        delete[] data;
    }
}
