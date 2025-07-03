// detector.h
#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#ifdef DETECTOR_DLL_EXPORTS
#define DETECTOR_API __declspec(dllexport)
#else
#define DETECTOR_API __declspec(dllimport)
#endif

struct DetectionResult {
    std::string class_name;
    cv::Rect bbox;
    float confidence;
    int area;
    std::vector<std::vector<cv::Point>> contours;
    float area_contour;
};
std::string detection_results_to_string(const std::vector<DetectionResult>& results);

class YOLO12Infer {
public:
    YOLO12Infer(const std::wstring& onnx_model, cv::Size input_image_size, float confidence_thres = 0.5, float iou_thres = 0.45);
    std::vector<DetectionResult> infer(const std::string& image_path);
    std::vector<DetectionResult> predict(const std::string& image_path, bool visual = false, bool show_score = true, bool show_class = true, bool save_or_not = false);
    void draw_box(cv::Mat& img, const DetectionResult& res, bool show_score, bool show_class);

private:
    cv::Mat letterbox(const cv::Mat& img, float& r, float& dw, float& dh, const cv::Scalar& color = cv::Scalar(114, 114, 114));
    bool preprocess(const std::string& image_path, std::vector<float>& input_tensor, cv::Mat& img, float& r, float& dw, float& dh);
    std::vector<DetectionResult> postprocess(const std::vector<float>& output, int num_boxes, int num_channels, const cv::Mat& img, float r, float dw, float dh);

    cv::Size input_image_size_;
    float confidence_thres_;
    float iou_thres_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
};