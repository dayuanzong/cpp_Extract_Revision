#include "FacePipeline.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <cstdio>
#include <algorithm>
#include <filesystem>
#include <onnxruntime_cxx_api.h>

// Constants
const float LANDMARKS_2D_NEW[] = {
    0.000213256f, 0.106454f, 0.0752622f, 0.038915f, 0.18113f, 0.0187482f, 0.29077f, 0.0344891f, 0.393397f, 0.0773906f,
    0.586856f, 0.0773906f, 0.689483f, 0.0344891f, 0.799124f, 0.0187482f, 0.904991f, 0.038915f, 0.98004f, 0.106454f,
    0.490127f, 0.203352f, 0.490127f, 0.307009f, 0.490127f, 0.409805f, 0.490127f, 0.515625f, 0.36688f, 0.587326f,
    0.426036f, 0.609345f, 0.490127f, 0.628106f, 0.554217f, 0.609345f, 0.613373f, 0.587326f, 0.121737f, 0.216423f,
    0.187122f, 0.178758f, 0.265825f, 0.179852f, 0.334606f, 0.231733f, 0.260918f, 0.245099f, 0.182743f, 0.244077f,
    0.645647f, 0.231733f, 0.714428f, 0.179852f, 0.793132f, 0.178758f, 0.858516f, 0.216423f, 0.79751f, 0.244077f,
    0.719335f, 0.245099f, 0.254149f, 0.780233f, 0.726104f, 0.780233f
};

const float LANDMARKS_68_3D[] = {
    -73.393523f, -29.801432f, 47.667532f,
    -72.775014f, -10.949766f, 45.909403f,
    -70.533638f, 7.929818f, 44.84258f,
    -66.850058f, 26.07428f, 43.141114f,
    -59.790187f, 42.56439f, 38.635298f,
    -48.368973f, 56.48108f, 30.750622f,
    -34.121101f, 67.246992f, 18.456453f,
    -17.875411f, 75.056892f, 3.609035f,
    0.098749f, 77.061286f, -0.881698f,
    17.477031f, 74.758448f, 5.181201f,
    32.648966f, 66.929021f, 19.176563f,
    46.372358f, 56.311389f, 30.77057f,
    57.34348f, 42.419126f, 37.628629f,
    64.388482f, 25.45588f, 40.886309f,
    68.212038f, 6.990805f, 42.281449f,
    70.486405f, -11.666193f, 44.142567f,
    71.375822f, -30.365191f, 47.140426f,
    -61.119406f, -49.361602f, 14.254422f,
    -51.287588f, -58.769795f, 7.268147f,
    -37.8048f, -61.996155f, 0.442051f,
    -24.022754f, -61.033399f, -6.606501f,
    -11.635713f, -56.686759f, -11.967398f,
    12.056636f, -57.391033f, -11.825605f,
    25.106256f, -61.902186f, -7.315098f,
    38.338588f, -62.777713f, -1.022953f,
    51.191007f, -59.302347f, 5.349435f,
    60.053851f, -50.190255f, 11.615746f,
    0.65394f, -42.19379f, -19.246394f,
    0.804809f, -30.993721f, -20.150396f,
    0.992204f, -19.944596f, -19.246394f,
    1.226783f, -8.414541f, -18.39558f,
    -14.772472f, 3.50195f, -22.89837f,
    -7.180239f, 4.251589f, -23.775597f,
    0.55592f, 4.464636f, -24.409778f,
    8.272499f, 4.251589f, -23.775597f,
    15.214351f, 3.50195f, -22.89837f,
    -46.04729f, -37.471411f, 7.037989f,
    -37.674688f, -42.73051f, 3.021217f,
    -27.883856f, -42.711517f, 1.353629f,
    -19.648268f, -36.754742f, -0.111088f,
    -28.272965f, -34.136564f, -0.147273f,
    -38.082418f, -34.919043f, 1.476612f,
    19.265868f, -36.754742f, -0.111088f,
    27.50075f, -42.711517f, 1.353629f,
    37.291073f, -42.73051f, 3.021217f,
    45.631856f, -37.471411f, 7.037989f,
    37.006216f, -34.919043f, 1.476612f,
    27.197145f, -34.136564f, -0.147273f,
    -28.979002f, -10.949766f, -10.815364f,
    -17.157617f, -12.759135f, -12.089354f,
    -4.200272f, -12.759135f, -12.089354f,
    8.550019f, -12.759135f, -12.089354f,
    20.371404f, -10.949766f, -10.815364f,
    8.272499f, -2.824512f, -14.005568f,
    -4.200272f, -2.824512f, -14.005568f,
    -15.184185f, -2.824512f, -13.226234f,
    -4.569668f, 7.426143f, -12.093049f,
    8.661375f, 7.426143f, -12.093049f,
    23.095216f, 7.426143f, -11.39182f,
    8.661375f, 17.563062f, -9.148217f,
    -4.569668f, 17.563062f, -9.148217f,
    -15.184185f, 17.563062f, -8.531513f
};

// FaceType Enum (Matching Python)
enum FaceType {
    HALF = 0,
    MID_FULL = 1,
    FULL = 2,
    FULL_NO_ALIGN = 3,
    WHOLE_FACE = 4,
    HEAD = 10,
    HEAD_NO_ALIGN = 20
};

static cv::Mat Umeyama(const std::vector<cv::Point2f>& src, const std::vector<cv::Point2f>& dst, bool estimate_scale);
static cv::Mat RotateForDetect(const cv::Mat& img, int rot) {
    if (rot == 0) return img;
    cv::Mat out;
    if (rot == 90) cv::rotate(img, out, cv::ROTATE_90_CLOCKWISE);
    else if (rot == 180) cv::rotate(img, out, cv::ROTATE_180);
    else if (rot == 270) cv::rotate(img, out, cv::ROTATE_90_COUNTERCLOCKWISE);
    else return img;
    return out;
}

static Face RotateRectBack(const Face& f, int rot, int w, int h) {
    if (rot == 0) return f;
    float l = f.x1, t = f.y1, r = f.x2, b = f.y2;
    float nl = l, nt = t, nr = r, nb = b;
    if (rot == 90) {
        nl = t;
        nt = h - l;
        nr = b;
        nb = h - r;
    } else if (rot == 180) {
        nl = w - l;
        nt = h - t;
        nr = w - r;
        nb = h - b;
    } else if (rot == 270) {
        nl = w - b;
        nt = l;
        nr = w - t;
        nb = r;
    }
    Face out = f;
    out.x1 = std::min(nl, nr);
    out.x2 = std::max(nl, nr);
    out.y1 = std::min(nt, nb);
    out.y2 = std::max(nt, nb);
    return out;
}

static std::vector<cv::Point2f> RotatePointsBack(const std::vector<cv::Point2f>& pts, int rot, int w, int h) {
    if (rot == 0) return pts;
    std::vector<cv::Point2f> out;
    out.reserve(pts.size());
    for (const auto& p : pts) {
        float x = p.x;
        float y = p.y;
        float nx = x;
        float ny = y;
        if (rot == 90) {
            nx = y;
            ny = h - x;
        } else if (rot == 180) {
            nx = w - x;
            ny = h - y;
        } else if (rot == 270) {
            nx = w - y;
            ny = x;
        }
        out.emplace_back(nx, ny);
    }
    return out;
}

class InsightFaceExtractor {
public:
    InsightFaceExtractor(const std::wstring& model_path, int device_id)
        : env(ORT_LOGGING_LEVEL_WARNING, "InsightFace") {
        Ort::SessionOptions session_options;

        if (device_id >= 0) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = device_id;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "InsightFace: CUDA Provider enabled on device " << device_id << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "InsightFace: Failed to enable CUDA Provider: " << e.what() << std::endl;
                std::cerr << "InsightFace: Falling back to CPU." << std::endl;
            }
        } else {
            std::cout << "InsightFace: CPU Mode requested." << std::endl;
        }

        session = Ort::Session(env, model_path.c_str(), session_options);

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

        auto out_info = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto out_shape = out_info.GetShape();
        if (!out_shape.empty() && out_shape.back() > 0) {
            emb_dim = static_cast<int>(out_shape.back());
        } else {
            emb_dim = 512;
        }
    }

    int GetEmbeddingDim() const {
        return emb_dim;
    }

    bool ExtractEmbedding(const cv::Mat& img, const std::vector<cv::Point2f>& landmarks, std::vector<float>& out) {
        if (img.empty() || landmarks.size() < 68) return false;

        cv::Mat aligned = AlignArcFace(img, landmarks);
        if (aligned.empty()) return false;

        std::vector<int64_t> input_shape = {1, 3, 112, 112};
        size_t input_tensor_size = 1 * 3 * 112 * 112;
        std::vector<float> input_tensor_values(input_tensor_size);

        for (int y = 0; y < 112; y++) {
            for (int x = 0; x < 112; x++) {
                cv::Vec3b pixel = aligned.at<cv::Vec3b>(y, x);
                float r = (pixel[2] - 127.5f) / 127.5f;
                float g = (pixel[1] - 127.5f) / 127.5f;
                float b = (pixel[0] - 127.5f) / 127.5f;
                size_t idx = y * 112 + x;
                input_tensor_values[idx] = r;
                input_tensor_values[112 * 112 + idx] = g;
                input_tensor_values[2 * 112 * 112 + idx] = b;
            }
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size());
        if (output_tensors.empty()) return false;

        float* data = output_tensors[0].GetTensorMutableData<float>();
        size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (!data || count == 0) return false;

        out.assign(data, data + count);

        float norm = 0.0f;
        for (size_t i = 0; i < out.size(); i++) {
            norm += out[i] * out[i];
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (size_t i = 0; i < out.size(); i++) {
                out[i] /= norm;
            }
        }
        return true;
    }

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    int emb_dim = 0;

    static cv::Mat AlignArcFace(const cv::Mat& img, const std::vector<cv::Point2f>& landmarks) {
        if (landmarks.size() < 68) return cv::Mat();
        cv::Point2f le(0, 0), re(0, 0);
        for (int i = 36; i <= 41; i++) le += landmarks[i];
        for (int i = 42; i <= 47; i++) re += landmarks[i];
        le *= (1.0f / 6.0f);
        re *= (1.0f / 6.0f);

        cv::Point2f nose = landmarks[30];
        cv::Point2f lm = landmarks[48];
        cv::Point2f rm = landmarks[54];

        std::vector<cv::Point2f> src = {le, re, nose, lm, rm};
        std::vector<cv::Point2f> dst = {
            {38.2946f, 51.6963f},
            {73.5318f, 51.5014f},
            {56.0252f, 71.7366f},
            {41.5493f, 92.3655f},
            {70.7299f, 92.2041f}
        };

        cv::Mat M = Umeyama(src, dst, true);
        if (M.empty()) return cv::Mat();
        cv::Mat aligned;
        cv::warpAffine(img, aligned, M, cv::Size(112, 112), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        return aligned;
    }
};

static std::wstring ResolveModelPath(const std::wstring& override_path, const std::vector<std::wstring>& fallback_paths) {
    if (!override_path.empty()) return override_path;
    for (const auto& p : fallback_paths) {
        if (std::filesystem::exists(p)) return p;
    }
    return fallback_paths.empty() ? std::wstring() : fallback_paths[0];
}

FacePipeline::FacePipeline(const std::wstring& model_dir, int device_id)
    : FacePipeline(model_dir, device_id, L"", L"", L"") {}

FacePipeline::FacePipeline(const std::wstring& model_dir, int device_id, const std::wstring& s3fd_override, const std::wstring& fan_override, const std::wstring& insight_override) {
    std::vector<std::wstring> s3fd_candidates = {
        model_dir + L"/S3FD/S3FD.onnx",
        model_dir + L"/S3FD.onnx"
    };
    std::vector<std::wstring> fan_candidates = {
        model_dir + L"/FAN/2DFAN-4.onnx",
        model_dir + L"/FAN/2DFAN.onnx",
        model_dir + L"/2DFAN-4.onnx",
        model_dir + L"/2DFAN.onnx"
    };
    std::vector<std::wstring> insight_candidates = {
        model_dir + L"/w600k_r50.onnx",
        model_dir + L"/insightface/w600k_r50.onnx"
    };

    std::wstring s3fd_path = ResolveModelPath(s3fd_override, s3fd_candidates);
    std::wstring fan_path = ResolveModelPath(fan_override, fan_candidates);
    std::wstring insight_path = ResolveModelPath(insight_override, insight_candidates);

    if (s3fd_path.find(L"det_10g") != std::wstring::npos || s3fd_path.find(L"scrfd") != std::wstring::npos) {
        scrfd = new SCRFDExtractor(s3fd_path, device_id, 640);
        std::wcout << L"Detector model: " << s3fd_path << std::endl;
        std::wcout << L"Detector type: SCRFD (det_10g), input_size=640" << std::endl;
    } else {
        s3fd = new S3FDExtractor(s3fd_path, device_id);
        std::wcout << L"Detector model: " << s3fd_path << std::endl;
        std::wcout << L"Detector type: S3FD, scale_to=640 if max_dim>=1280 else max_dim/2, min=64" << std::endl;
    }

    if (fan_path.find(L"2d106det") != std::wstring::npos || fan_path.find(L"1k3d68") != std::wstring::npos) {
        insight_landmark_is_3d = (fan_path.find(L"1k3d68") != std::wstring::npos);
        insight_landmark = new InsightFaceLandmark(fan_path, device_id);
    } else {
        fan = new FANExtractor(fan_path, device_id);
    }

    if (!insight_path.empty() && std::filesystem::exists(insight_path)) {
        insight = new InsightFaceExtractor(insight_path, device_id);
    }
}

FacePipeline::~FacePipeline() {
    if (s3fd) delete s3fd;
    if (scrfd) delete scrfd;
    if (fan) delete fan;
    if (insight_landmark) delete insight_landmark;
    if (insight) delete insight;
}

void FacePipeline::SetFilterParams(bool enable_blur_value, float blur_low_value, float blur_high_value,
                                   bool enable_pose_value, float pitch_threshold_value, float yaw_threshold_value,
                                   bool enable_mouth_value, float mouth_threshold_value) {
    enable_blur = enable_blur_value;
    blur_low = blur_low_value;
    blur_high = blur_high_value;
    enable_pose = enable_pose_value;
    pitch_threshold = pitch_threshold_value;
    yaw_threshold = yaw_threshold_value;
    enable_mouth = enable_mouth_value;
    mouth_threshold = mouth_threshold_value;
}

void FacePipeline::SetAlignSize(int size) {
    if (size > 0) {
        align_size = size;
    }
}

void FacePipeline::SetMaxFaces(int max_faces_value) {
    max_faces = max_faces_value;
}

void FacePipeline::SetJpegQuality(int quality) {
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;
    jpeg_quality = quality;
}

int FacePipeline::GetEmbeddingDim() const {
    if (!insight) return 0;
    return insight->GetEmbeddingDim();
}

bool FacePipeline::ExtractEmbedding(const cv::Mat& img, const std::vector<cv::Point2f>& landmarks, std::vector<float>& out) {
    if (!insight) return false;
    return insight->ExtractEmbedding(img, landmarks, out);
}

void FacePipeline::SetReferenceEmbeddings(const float* refs, int ref_count, int ref_dim, float sim_threshold) {
    reference_embeddings.clear();
    reference_count = 0;
    reference_dim = 0;
    reference_threshold = sim_threshold;
    if (!refs || ref_count <= 0 || ref_dim <= 0) return;
    reference_embeddings.assign(refs, refs + (size_t)ref_count * (size_t)ref_dim);
    reference_count = ref_count;
    reference_dim = ref_dim;
    for (int i = 0; i < reference_count; i++) {
        float norm = 0.0f;
        float* base = reference_embeddings.data() + (size_t)i * (size_t)reference_dim;
        for (int j = 0; j < reference_dim; j++) {
            norm += base[j] * base[j];
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (int j = 0; j < reference_dim; j++) {
                base[j] /= norm;
            }
        }
    }
}

void FacePipeline::ClearReferenceEmbeddings() {
    reference_embeddings.clear();
    reference_count = 0;
    reference_dim = 0;
}

void FacePipeline::FreeFaceInfo(FaceInfo& info) {
    if (info.jpg_data) {
        delete[] info.jpg_data;
        info.jpg_data = nullptr;
    }
}

static double Det2x2(const cv::Mat& m) {
    return m.at<double>(0, 0) * m.at<double>(1, 1) - m.at<double>(0, 1) * m.at<double>(1, 0);
}

static cv::Mat Umeyama(const std::vector<cv::Point2f>& src, const std::vector<cv::Point2f>& dst, bool estimate_scale) {
    if (src.size() != dst.size() || src.empty()) return cv::Mat();

    int num = (int)src.size();
    int dim = 2;

    cv::Point2d src_mean(0, 0), dst_mean(0, 0);
    for (int i = 0; i < num; ++i) {
        src_mean.x += src[i].x;
        src_mean.y += src[i].y;
        dst_mean.x += dst[i].x;
        dst_mean.y += dst[i].y;
    }
    src_mean.x /= num;
    src_mean.y /= num;
    dst_mean.x /= num;
    dst_mean.y /= num;

    cv::Mat src_demean(num, dim, CV_64F);
    cv::Mat dst_demean(num, dim, CV_64F);
    for (int i = 0; i < num; ++i) {
        src_demean.at<double>(i, 0) = src[i].x - src_mean.x;
        src_demean.at<double>(i, 1) = src[i].y - src_mean.y;
        dst_demean.at<double>(i, 0) = dst[i].x - dst_mean.x;
        dst_demean.at<double>(i, 1) = dst[i].y - dst_mean.y;
    }

    cv::Mat A = (dst_demean.t() * src_demean) / (double)num;

    double d0 = 1.0;
    double d1 = 1.0;
    if (cv::determinant(A) < 0.0) d1 = -1.0;

    cv::SVD svd(A);
    cv::Mat U = svd.u;
    cv::Mat Vt = svd.vt;

    int rank = cv::countNonZero(svd.w > 1e-9);
    if (rank == 0) {
        cv::Mat nan_mat = cv::Mat::ones(2, 3, CV_64F) * std::numeric_limits<double>::quiet_NaN();
        return nan_mat;
    }

    cv::Mat D = cv::Mat::eye(2, 2, CV_64F);
    D.at<double>(0, 0) = d0;
    D.at<double>(1, 1) = d1;

    cv::Mat R;
    if (rank == dim - 1) {
        if (Det2x2(U) * Det2x2(Vt.t()) > 0) {
            R = U * Vt;
        } else {
            double tmp = D.at<double>(1, 1);
            D.at<double>(1, 1) = -1.0;
            R = U * D * Vt;
            D.at<double>(1, 1) = tmp;
        }
    } else {
        R = U * D * Vt;
    }

    double scale = 1.0;
    if (estimate_scale) {
        double var = 0.0;
        for (int i = 0; i < num; ++i) {
            double x = src_demean.at<double>(i, 0);
            double y = src_demean.at<double>(i, 1);
            var += x * x + y * y;
        }
        var /= (double)num;
        double s0 = svd.w.at<double>(0, 0);
        double s1 = svd.w.at<double>(1, 0);
        scale = (s0 * d0 + s1 * d1) / var;
    }

    cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat Rs = R * scale;
    Rs.copyTo(T(cv::Rect(0, 0, 2, 2)));

    cv::Mat src_mean_mat = (cv::Mat_<double>(2, 1) << src_mean.x, src_mean.y);
    cv::Mat dst_mean_mat = (cv::Mat_<double>(2, 1) << dst_mean.x, dst_mean.y);
    cv::Mat trans = dst_mean_mat - Rs * src_mean_mat;
    T.at<double>(0, 2) = trans.at<double>(0, 0);
    T.at<double>(1, 2) = trans.at<double>(1, 0);

    return T(cv::Rect(0, 0, 3, 2)).clone();
}

std::vector<cv::Point2f> TransformPoints(const std::vector<cv::Point2f>& points, const cv::Mat& M) {
    std::vector<cv::Point2f> dst;
    cv::transform(points, dst, M);
    return dst;
}

std::vector<cv::Point2f> InverseTransformPoints(const std::vector<cv::Point2f>& points, const cv::Mat& M) {
    cv::Mat M_inv;
    cv::invertAffineTransform(M, M_inv);
    std::vector<cv::Point2f> dst;
    cv::transform(points, dst, M_inv);
    return dst;
}

static double PolygonArea(const std::vector<cv::Point2f>& pts) {
    double area = 0.0;
    int n = (int)pts.size();
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += (double)pts[i].x * pts[j].y;
        area -= (double)pts[i].y * pts[j].x;
    }
    return std::abs(area) * 0.5;
}

static double EstimateAveragedYaw(const std::vector<cv::Point2f>& landmarks) {
    double l = ((landmarks[27].x - landmarks[0].x) + (landmarks[28].x - landmarks[1].x) + (landmarks[29].x - landmarks[2].x)) / 3.0;
    double r = ((landmarks[16].x - landmarks[27].x) + (landmarks[15].x - landmarks[28].x) + (landmarks[14].x - landmarks[29].x)) / 3.0;
    return r - l;
}

static std::vector<cv::Point2f> Convert106To68(const std::vector<cv::Point2f>& lm106) {
    if (lm106.size() < 106) return {};
    static const int mapping[68] = {
        1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17,
        43, 48, 49, 51, 50,
        102, 103, 104, 105, 101,
        72, 73, 74, 86, 78, 79, 80, 85, 84,
        35, 41, 42, 39, 37, 36,
        89, 95, 96, 93, 91, 90,
        52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70, 69, 57, 60, 54
    };
    std::vector<cv::Point2f> out;
    out.reserve(68);
    for (int i = 0; i < 68; i++) {
        int idx = mapping[i];
        if (idx >= 0 && idx < (int)lm106.size()) out.push_back(lm106[idx]);
        else out.push_back(cv::Point2f(0, 0));
    }
    return out;
}

static void NormalizeLandmarkOrientation(std::vector<cv::Point2f>& lms, const cv::Rect2f& rect) {
    if (lms.size() != 68) return;
    cv::Point2f le(0, 0), re(0, 0), mouth(0, 0);
    for (int i = 36; i <= 41; i++) le += lms[i];
    for (int i = 42; i <= 47; i++) re += lms[i];
    for (int i = 48; i <= 67; i++) mouth += lms[i];
    le *= (1.0f / 6.0f);
    re *= (1.0f / 6.0f);
    mouth *= (1.0f / 20.0f);
    float cx = rect.x + rect.width * 0.5f;
    float cy = rect.y + rect.height * 0.5f;
    if (le.x > re.x) {
        for (auto& p : lms) p.x = 2.0f * cx - p.x;
        std::swap(le, re);
    }
    float eye_y = (le.y + re.y) * 0.5f;
    if (mouth.y < eye_y) {
        for (auto& p : lms) p.y = 2.0f * cy - p.y;
    }
}

cv::Mat FacePipeline::GetTransformMat(const std::vector<cv::Point2f>& src_points, int size, int face_type) {
    std::vector<cv::Point2f> umeyama_src;
    std::vector<cv::Point2f> umeyama_dst;
    for (int i = 17; i < 49; i++) {
        cv::Point2f p = src_points[i];
        if (std::abs(p.x) > 1e-4 || std::abs(p.y) > 1e-4) {
            int dst_idx = i - 17;
            umeyama_src.push_back(p);
            umeyama_dst.push_back(cv::Point2f(LANDMARKS_2D_NEW[2 * dst_idx], LANDMARKS_2D_NEW[2 * dst_idx + 1]));
        }
    }
    
    cv::Point2f p54 = src_points[54];
    if (std::abs(p54.x) > 1e-4 || std::abs(p54.y) > 1e-4) {
        int dst_idx = 32;
        umeyama_src.push_back(p54);
        umeyama_dst.push_back(cv::Point2f(LANDMARKS_2D_NEW[2 * dst_idx], LANDMARKS_2D_NEW[2 * dst_idx + 1]));
    }

    if (umeyama_src.size() < 3) return cv::Mat();

    cv::Mat M = Umeyama(umeyama_src, umeyama_dst, true);

    std::vector<cv::Point2f> norm_corners = {
        {0,0}, {1,0}, {1,1}, {0,1}, {0.5f, 0.5f}
    };
    
    std::vector<cv::Point2f> g_p = InverseTransformPoints(norm_corners, M);
    cv::Point2f g_c = g_p[4];

    cv::Point2f tb_diag_vec = g_p[2] - g_p[0];
    float tb_norm = (float)cv::norm(tb_diag_vec);
    tb_diag_vec /= tb_norm;

    cv::Point2f bt_diag_vec = g_p[1] - g_p[3];
    float bt_norm = (float)cv::norm(bt_diag_vec);
    bt_diag_vec /= bt_norm;

    float padding = 0.0f;
    bool remove_align = false;
    
    switch (face_type) {
        case HALF: padding = 0.0f; break;
        case MID_FULL: padding = 0.0675f; break;
        case FULL: padding = 0.2109375f; break;
        case FULL_NO_ALIGN: padding = 0.2109375f; remove_align = true; break;
        case WHOLE_FACE: padding = 0.40f; break;
        case HEAD: padding = 0.70f; break;
        case HEAD_NO_ALIGN: padding = 0.70f; remove_align = true; break;
        default: padding = 0.2109375f; break;
    }

    float scale = 1.0f;
    float mod = (1.0f / scale) * (cv::norm(g_p[0] - g_p[2]) * (padding * sqrt(2.0f) + 0.5f));

    if (face_type == WHOLE_FACE) {
        cv::Point2f vec = g_p[0] - g_p[3];
        float vec_len = (float)cv::norm(vec);
        vec /= vec_len;
        g_c += vec * vec_len * 0.07f;
    } else if (face_type == HEAD) {
        auto aligned = TransformPoints(src_points, M);
        double yaw = EstimateAveragedYaw(aligned);
        yaw *= std::abs(std::tanh(yaw * 2.0));
        cv::Point2f hvec = g_p[0] - g_p[1];
        float hvec_len = (float)cv::norm(hvec);
        hvec /= hvec_len;
        g_c -= hvec * (float)(yaw * hvec_len / 2.0);
        cv::Point2f vvec = g_p[0] - g_p[3];
        float vvec_len = (float)cv::norm(vvec);
        vvec /= vvec_len;
        g_c += vvec * vvec_len * 0.50f;
    }

    // calc 3 points in global space to estimate 2d affine transform
    std::vector<cv::Point2f> src_tri;
    
    if (!remove_align) {
        src_tri.push_back(g_c - tb_diag_vec * mod);
        src_tri.push_back(g_c + bt_diag_vec * mod);
        src_tri.push_back(g_c + tb_diag_vec * mod);
    } else {
        src_tri.push_back(g_c - tb_diag_vec * mod);
        src_tri.push_back(g_c + bt_diag_vec * mod);
        src_tri.push_back(g_c + tb_diag_vec * mod);
        src_tri.push_back(g_c - bt_diag_vec * mod);

        double area = PolygonArea(src_tri);
        float side = (float)(std::sqrt(area) / 2.0);

        src_tri.clear();
        src_tri.push_back(g_c + cv::Point2f(-side, -side));
        src_tri.push_back(g_c + cv::Point2f(side, -side));
        src_tri.push_back(g_c + cv::Point2f(side, side));
    }

    std::vector<cv::Point2f> dst_tri = {
        {0, 0}, {(float)size, 0}, {(float)size, (float)size}
    };

    return cv::getAffineTransform(src_tri, dst_tri);
}

cv::Mat FacePipeline::AlignFace(const cv::Mat& img, const std::vector<cv::Point2f>& landmarks, int size, int face_type) {
    cv::Mat M = GetTransformMat(landmarks, size, face_type);
    cv::Mat aligned;
    cv::warpAffine(img, aligned, M, cv::Size(size, size), cv::INTER_LANCZOS4);
    return aligned;
}

static std::vector<cv::Point> ExpandEyebrows(const std::vector<cv::Point2f>& lmrks, float mod) {
    if (lmrks.size() != 68) return {};
    std::vector<cv::Point> result(68);
    for (int i = 0; i < 68; i++) {
        result[i] = cv::Point((int)lmrks[i].x, (int)lmrks[i].y);
    }

    cv::Point ml_pnt((result[36].x + result[0].x) / 2, (result[36].y + result[0].y) / 2);
    cv::Point mr_pnt((result[16].x + result[45].x) / 2, (result[16].y + result[45].y) / 2);

    cv::Point ql_pnt((result[36].x + ml_pnt.x) / 2, (result[36].y + ml_pnt.y) / 2);
    cv::Point qr_pnt((result[45].x + mr_pnt.x) / 2, (result[45].y + mr_pnt.y) / 2);

    std::vector<cv::Point> bot_l = { ql_pnt, result[36], result[37], result[38], result[39] };
    std::vector<cv::Point> bot_r = { result[42], result[43], result[44], result[45], qr_pnt };
    std::vector<cv::Point> top_l(result.begin() + 17, result.begin() + 22);
    std::vector<cv::Point> top_r(result.begin() + 22, result.begin() + 27);

    for (int i = 0; i < 5; i++) {
        float nx = top_l[i].x + mod * 0.5f * (top_l[i].x - bot_l[i].x);
        float ny = top_l[i].y + mod * 0.5f * (top_l[i].y - bot_l[i].y);
        result[17 + i] = cv::Point((int)nx, (int)ny);
    }
    for (int i = 0; i < 5; i++) {
        float nx = top_r[i].x + mod * 0.5f * (top_r[i].x - bot_r[i].x);
        float ny = top_r[i].y + mod * 0.5f * (top_r[i].y - bot_r[i].y);
        result[22 + i] = cv::Point((int)nx, (int)ny);
    }

    return result;
}

static cv::Mat GetImageHullMask(const cv::Size& size, const std::vector<cv::Point2f>& landmarks, float eyebrows_expand_mod) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8U);
    std::vector<cv::Point> lmrks = ExpandEyebrows(landmarks, eyebrows_expand_mod);
    if (lmrks.size() != 68) return mask;

    auto append_range = [&](std::vector<cv::Point>& dst, int start, int end) {
        for (int i = start; i < end; i++) dst.push_back(lmrks[i]);
    };

    auto fill_part = [&](std::initializer_list<std::pair<int, int>> ranges) {
        std::vector<cv::Point> merged;
        for (const auto& r : ranges) append_range(merged, r.first, r.second);
        if (merged.size() < 3) return;
        std::vector<cv::Point> hull;
        cv::convexHull(merged, hull);
        if (hull.size() < 3) return;
        cv::fillConvexPoly(mask, hull, cv::Scalar(255));
    };

    fill_part({ {0, 9}, {17, 18} });
    fill_part({ {8, 17}, {26, 27} });
    fill_part({ {17, 20}, {8, 9} });
    fill_part({ {24, 27}, {8, 9} });
    fill_part({ {19, 25}, {8, 9} });
    fill_part({ {17, 22}, {27, 28}, {31, 36}, {8, 9} });
    fill_part({ {22, 27}, {27, 28}, {31, 36}, {8, 9} });
    fill_part({ {27, 31}, {31, 36} });

    return mask;
}

float FacePipeline::ComputeBlur(const cv::Mat& img, const std::vector<cv::Point2f>& landmarks) {
    if (img.empty() || landmarks.empty()) return -1.0f;
    cv::Mat gray;
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else gray = img.clone();

    cv::Mat mask = GetImageHullMask(gray.size(), landmarks, 1.0f);
    if (mask.empty()) return -1.0f;
    if (cv::countNonZero(mask) == 0) return -1.0f;

    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev, mask);
    return (float)(stddev[0] * stddev[0]);
}

static cv::Vec3d RotationMatrixToEuler(const cv::Mat& R) {
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular) {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return cv::Vec3d(x, y, z);
}

std::string FacePipeline::EstimatePose(const std::vector<cv::Point2f>& landmarks, float& pitch, float& yaw, float& roll) {
    pitch = 0.0f;
    yaw = 0.0f;
    roll = 0.0f;
    if (landmarks.size() < 68) return "Unknown";

    std::vector<cv::Point3f> model_points;
    std::vector<cv::Point2f> image_points;
    model_points.reserve(33);
    image_points.reserve(33);

    auto push_idx = [&](int idx) {
        model_points.emplace_back(LANDMARKS_68_3D[idx * 3 + 0], LANDMARKS_68_3D[idx * 3 + 1], LANDMARKS_68_3D[idx * 3 + 2]);
        image_points.emplace_back(landmarks[idx]);
    };

    for (int i = 0; i < 27; i++) push_idx(i);
    for (int i = 30; i < 36; i++) push_idx(i);

    float size = (float)align_size;
    cv::Mat camera_matrix = (cv::Mat_<float>(3, 3) <<
        size, 0.0f, size / 2.0f,
        0.0f, size, size / 2.0f,
        0.0f, 0.0f, 1.0f);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_32F);
    cv::Mat rvec, tvec;

    bool ok = cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    if (!ok) return "Unknown";

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Vec3d euler = RotationMatrixToEuler(R);

    pitch = (float)(-euler[0]);
    yaw = (float)euler[1];
    roll = (float)euler[2];

    float half_pi = 1.57079632679f;
    pitch = std::max(-half_pi, std::min(half_pi, pitch));
    yaw = std::max(-half_pi, std::min(half_pi, yaw));
    roll = std::max(-half_pi, std::min(half_pi, roll));

    float pitch_deg = pitch * 57.29578f;
    float yaw_deg = yaw * 57.29578f;

    if (pitch_deg > pitch_threshold) return "抬头";
    if (pitch_deg < -pitch_threshold) return "低头";
    if (yaw_deg < -yaw_threshold) return "向右";
    if (yaw_deg > yaw_threshold) return "向左";
    return "Unknown";
}

bool FacePipeline::IsMouthOpen(const std::vector<cv::Point2f>& landmarks, float threshold) {
    if (landmarks.size() < 67) return false;
    return (landmarks[66].y - landmarks[62].y) > threshold;
}

static float GetMouthValue(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() < 67) return 0.0f;
    return landmarks[66].y - landmarks[62].y;
}

std::vector<FaceInfo> FacePipeline::Process(const std::wstring& img_path, int face_type) {
    std::vector<FaceInfo> results;
#ifdef _WIN32
    FILE* f = _wfopen(img_path.c_str(), L"rb");
    if (!f) return results;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uchar> buf(sz);
    fread(buf.data(), 1, sz, f);
    fclose(f);
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
#else
    std::string path_utf8(img_path.begin(), img_path.end());
    cv::Mat img = cv::imread(path_utf8);
#endif
    if (img.empty()) return results;
    return ProcessMat(img, face_type);
}

std::vector<FaceInfo> FacePipeline::ProcessMat(const cv::Mat& img, int face_type) {
    std::vector<FaceInfo> results;
    if (img.empty()) return results;

    cv::Mat work = img;
    if (work.channels() == 1) {
        cv::cvtColor(work, work, cv::COLOR_GRAY2BGR);
    }
    int even_w = work.cols - (work.cols % 2);
    int even_h = work.rows - (work.rows % 2);
    if (even_w != work.cols || even_h != work.rows) {
        work = work(cv::Rect(0, 0, even_w, even_h)).clone();
    }

    if (std::min(work.rows, work.cols) < 128) {
        return results;
    }

    int used_rot = 0;
    cv::Mat detect_img = work;
    std::vector<Face> faces;
    if (s3fd) faces = s3fd->Detect(detect_img);
    else if (scrfd) faces = scrfd->Detect(detect_img);
    
    if (faces.empty()) {
        std::vector<int> rotations = {90, 270, 180};
        for (int rot : rotations) {
            cv::Mat rotated = RotateForDetect(work, rot);
            std::vector<Face> faces_rot;
            if (s3fd) faces_rot = s3fd->Detect(rotated);
            else if (scrfd) faces_rot = scrfd->Detect(rotated);
            
            if (!faces_rot.empty()) {
                used_rot = rot;
                detect_img = rotated;
                faces = std::move(faces_rot);
                break;
            }
        }
    }

    if (max_faces > 0 && (int)faces.size() > max_faces) {
        faces.resize((size_t)max_faces);
    }

    int w = work.cols;
    int h = work.rows;
    for (const auto& face : faces) {
        cv::Rect2f rect(face.x1, face.y1, face.x2 - face.x1, face.y2 - face.y1);
        std::vector<cv::Point2f> landmarks;
        if (fan) {
            landmarks = fan->Extract(detect_img, rect, true);
        } else if (insight_landmark) {
            auto lm = insight_landmark->Extract(detect_img, rect);
            if (lm.size() == 68) {
                landmarks = std::move(lm);
            } else if (lm.size() >= 106) {
                landmarks = Convert106To68(lm);
            }
            if (landmarks.size() == 68) {
                NormalizeLandmarkOrientation(landmarks, rect);
            }
        }

        if (landmarks.size() != 68) {
            continue;
        }

        cv::Mat M_second = GetTransformMat(landmarks, 256, FaceType::FULL);
        cv::Mat aligned_second;
        cv::warpAffine(detect_img, aligned_second, M_second, cv::Size(256, 256), cv::INTER_CUBIC);
        
        std::vector<Face> faces_second;
        if (s3fd) faces_second = s3fd->Detect(aligned_second);
        else if (scrfd) faces_second = scrfd->Detect(aligned_second);
        
        if (faces_second.size() == 1) {
            cv::Rect2f rect2(faces_second[0].x1, faces_second[0].y1, faces_second[0].x2 - faces_second[0].x1, faces_second[0].y2 - faces_second[0].y1);
            std::vector<cv::Point2f> lmrks2;
            if (fan) lmrks2 = fan->Extract(aligned_second, rect2, true);
            else if (insight_landmark) {
                 auto lm2 = insight_landmark->Extract(aligned_second, rect2);
                 if (lm2.size() == 68) {
                     lmrks2 = std::move(lm2);
                 } else if (lm2.size() >= 106) {
                     lmrks2 = Convert106To68(lm2);
                 }
                 if (lmrks2.size() == 68) {
                     NormalizeLandmarkOrientation(lmrks2, rect2);
                 }
            }
            
            if (lmrks2.size() == 68) {
                landmarks = InverseTransformPoints(lmrks2, M_second);
            }
        }

        Face face_orig = used_rot == 0 ? face : RotateRectBack(face, used_rot, w, h);
        if (used_rot != 0) {
            landmarks = RotatePointsBack(landmarks, used_rot, w, h);
        }

        int size = align_size;
        if (size <= 0) size = 256;
        cv::Mat M = GetTransformMat(landmarks, size, face_type);
        cv::Mat aligned;
        cv::warpAffine(work, aligned, M, cv::Size(size, size), cv::INTER_LANCZOS4);

        std::vector<cv::Point2f> aligned_lms;
        cv::transform(landmarks, aligned_lms, M);

        if (face_type <= FULL_NO_ALIGN && !(insight_landmark && insight_landmark_is_3d)) {
            float max_xy = (float)(size - 1);
            std::vector<cv::Point2f> aligned_corners = {
                {0,0}, {max_xy,0}, {max_xy,max_xy}, {0,max_xy}
            };
            cv::Mat M_inv;
            cv::invertAffineTransform(M, M_inv);
            std::vector<cv::Point2f> source_box_pts;
            cv::transform(aligned_corners, source_box_pts, M_inv);
            float landmarks_area = 0.0f;
            for(int i=0; i<4; i++) {
                int j = (i+1)%4;
                landmarks_area += source_box_pts[i].x * source_box_pts[j].y;
                landmarks_area -= source_box_pts[i].y * source_box_pts[j].x;
            }
            landmarks_area = std::abs(landmarks_area) * 0.5f;
            float rect_area = (face_orig.x2 - face_orig.x1) * (face_orig.y2 - face_orig.y1);
            
            if (landmarks_area > 4.0f * rect_area) {
                continue;
            }
        }

        std::vector<uchar> jpg_buf;
        cv::imencode(".jpg", aligned, jpg_buf, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality});

        FaceInfo info;
        info.jpg_size = (int)jpg_buf.size();
        info.jpg_data = new unsigned char[info.jpg_size];
        memcpy(info.jpg_data, jpg_buf.data(), info.jpg_size);

        // Copy source landmarks
        for (int i = 0; i < 68; i++) {
            info.landmarks[2*i] = landmarks[i].x;
            info.landmarks[2*i+1] = landmarks[i].y;
        }

        // Copy aligned landmarks
        for (int i = 0; i < 68; i++) {
            info.aligned_landmarks[2*i] = aligned_lms[i].x;
            info.aligned_landmarks[2*i+1] = aligned_lms[i].y;
        }

        info.embedding_dim = 0;
        info.target_index = -1;
        info.target_sim = -1.0f;
        info.is_target = false;
        for (int i = 0; i < 512; i++) {
            info.embedding[i] = 0.0f;
        }
        if (insight) {
            std::vector<float> emb;
            if (insight->ExtractEmbedding(work, landmarks, emb)) {
                int emb_dim = insight->GetEmbeddingDim();
                if (emb_dim <= 0) {
                    emb_dim = (int)emb.size();
                }
                int copy_dim = std::min(emb_dim, (int)emb.size());
                if (copy_dim > 512) copy_dim = 512;
                info.embedding_dim = copy_dim;
                for (int i = 0; i < copy_dim; i++) {
                    info.embedding[i] = emb[i];
                }
            }
        }
        if (reference_count > 0 && reference_dim > 0 && info.embedding_dim == reference_dim) {
            float best_sim = -1.0f;
            int best_idx = -1;
            for (int i = 0; i < reference_count; i++) {
                float sim = 0.0f;
                const float* ref = reference_embeddings.data() + (size_t)i * (size_t)reference_dim;
                for (int j = 0; j < reference_dim; j++) {
                    sim += ref[j] * info.embedding[j];
                }
                if (sim > best_sim) {
                    best_sim = sim;
                    best_idx = i;
                }
            }
            info.target_index = best_idx;
            info.target_sim = best_sim;
            info.is_target = (best_sim >= reference_threshold);
        }

        // Copy rect
        info.source_rect[0] = face_orig.x1;
        info.source_rect[1] = face_orig.y1;
        info.source_rect[2] = face_orig.x2;
        info.source_rect[3] = face_orig.y2;

        info.detect_score = face.score;
        info.blur_variance = ComputeBlur(aligned, aligned_lms);
        info.blur_class = -1;
        if (enable_blur && info.blur_variance >= 0.0f) {
            if (info.blur_variance < blur_low) info.blur_class = 0;
            else if (info.blur_variance < blur_high) info.blur_class = 1;
            else info.blur_class = 2;
        }

        info.pitch = 0.0f;
        info.yaw = 0.0f;
        info.roll = 0.0f;
        std::string pose_tag = "Unknown";
        if (enable_pose) {
            pose_tag = EstimatePose(aligned_lms, info.pitch, info.yaw, info.roll);
        }

        std::snprintf(info.pose_tag, sizeof(info.pose_tag), "%s", pose_tag.c_str());
        info.mouth_value = GetMouthValue(aligned_lms);
        info.mouth_open = enable_mouth ? info.mouth_value > mouth_threshold : false;
        info.valid = true;

        results.push_back(info);
    }

    return results;
}
