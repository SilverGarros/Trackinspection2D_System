// detector.cpp
#include "pch.h"
#include "Detector.h"
#include <filesystem>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
namespace fs = std::filesystem;

static const std::vector<std::string> CLASS_NAMES = { "DK", "ABC", "XYZ" };
static const std::map<std::string, cv::Scalar> CLASS_COLORS = {
    {"DK",  cv::Scalar(0, 0, 255)},
    {"ABC", cv::Scalar(0, 125, 0)},
    {"XYZ", cv::Scalar(255, 0, 0)}
};

std::string detection_results_to_string(const std::vector<DetectionResult>& results) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < results.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "{'class_name': '" << results[i].class_name << "'";
        oss << ", 'bbox': [" << results[i].bbox.x << ", " << results[i].bbox.y << ", " << results[i].bbox.width << ", " << results[i].bbox.height << "]";
        oss << ", 'confidence': " << results[i].confidence;
        oss << ", 'area': " << results[i].area;
        oss << "}";
    }
    oss << "]";
    return oss.str();
}

YOLO12Infer::YOLO12Infer(const std::wstring& onnx_model, cv::Size input_image_size, float confidence_thres, float iou_thres)
    : input_image_size_(input_image_size),
    confidence_thres_(confidence_thres),
    iou_thres_(iou_thres),
    env_(ORT_LOGGING_LEVEL_WARNING, "YOLO12"),
    session_options_(),
    session_(nullptr)
{
    session_options_.SetIntraOpNumThreads(1);
#ifdef _WIN32
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0);
#endif
    session_ = Ort::Session(env_, onnx_model.c_str(), session_options_);
}

cv::Mat YOLO12Infer::letterbox(const cv::Mat& img, float& r, float& dw, float& dh, const cv::Scalar& color) {
    int w = img.cols, h = img.rows;
    r = std::min(static_cast<float>(input_image_size_.width) / w, static_cast<float>(input_image_size_.height) / h);
    int new_unpad_w = static_cast<int>(round(w * r));
    int new_unpad_h = static_cast<int>(round(h * r));
    dw = (static_cast<float>(input_image_size_.width) - new_unpad_w) / 2.0f;
    dh = (static_cast<float>(input_image_size_.height) - new_unpad_h) / 2.0f;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_unpad_w, new_unpad_h));
    cv::Mat out;
    cv::copyMakeBorder(resized, out,
        static_cast<int>(std::floor(dh)), static_cast<int>(std::ceil(dh)),
        static_cast<int>(std::floor(dw)), static_cast<int>(std::ceil(dw)),
        cv::BORDER_CONSTANT, color);
    return out;
}

bool YOLO12Infer::preprocess(const std::string& image_path, std::vector<float>& input_tensor, cv::Mat& img, float& r, float& dw, float& dh) {
    img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) return false;
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    cv::Mat img_resized = letterbox(img_rgb, r, dw, dh);
    img_resized.convertTo(img_resized, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(img_resized, channels);
    // CHW
    for (int c = 0; c < 3; ++c) {
        input_tensor.insert(input_tensor.end(), (float*)channels[c].datastart, (float*)channels[c].dataend);
    }
    return true;
}

std::vector<DetectionResult> YOLO12Infer::postprocess(const std::vector<float>& output, int num_boxes, int num_channels, const cv::Mat& img, float r, float dw, float dh) {
    std::vector<DetectionResult> results;
    int img_w = img.cols, img_h = img.rows;

    for (int i = 0; i < num_boxes; ++i) {
        float x = output[i * num_channels + 0];
        float y = output[i * num_channels + 1];
        float w = output[i * num_channels + 2];
        float h = output[i * num_channels + 3];
        float obj_conf = output[i * num_channels + 4];

        float max_score = 0.0f;
        int class_id = -1;
        for (int c = 0; c < (int)CLASS_NAMES.size(); ++c) {
            float score = output[i * num_channels + 5 + c];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }
        float conf = obj_conf * max_score;
        if (conf < confidence_thres_ || class_id < 0) continue;
        std::cout << "raw: obj_conf=" << obj_conf << ", max_score=" << max_score << ", conf=" << conf << std::endl;
        // letterbox逆变换
        float x0 = (x - dw) / r;
        float y0 = (y - dh) / r;
        float ww = w / r;
        float hh = h / r;
        int left = static_cast<int>(x0 - ww / 2.0f + 0.5f);
        int top = static_cast<int>(y0 - hh / 2.0f + 0.5f);
        int width = static_cast<int>(ww + 0.5f);
        int height = static_cast<int>(hh + 0.5f);

        left = std::max(0, left);
        top = std::max(0, top);
        if (left + width > img.cols) width = img.cols - left;
        if (top + height > img.rows) height = img.rows - top;
        if (width <= 0 || height <= 0) continue;

        DetectionResult res;
        res.class_name = CLASS_NAMES[class_id];
        res.bbox = cv::Rect(left, top, width, height);
        res.confidence = conf;
        res.area = width * height;

        // 裁剪区域
        cv::Rect roi(left, top, width, height);
        cv::Mat crop = img(roi).clone();

        // 灰度化+反色+自适应阈值
        cv::Mat gray, gray_inv, binary;
        if (crop.channels() == 3)
            cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
        else
            gray = crop;
        cv::bitwise_not(gray, gray_inv);
        cv::adaptiveThreshold(gray_inv, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, -2);

        // 轮廓提取与面积
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        double area_contour = 0.0;
        for (auto& cnt : contours) {
            double a = cv::contourArea(cnt);
            if (a > 20) area_contour += a;
        }
        if (area_contour > res.area) area_contour = res.area;
        res.contours = contours;
        res.area_contour = static_cast<float>(area_contour);

        results.push_back(res);
    }

    // NMS
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (const auto& r : results) {
        boxes.push_back(r.bbox);
        scores.push_back(r.confidence);
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, confidence_thres_, iou_thres_, indices);

    std::vector<DetectionResult> nms_results;
    for (int idx : indices) {
        nms_results.push_back(results[idx]);
    }
    return nms_results;
}

std::vector<DetectionResult> YOLO12Infer::infer(const std::string& image_path) {
    std::vector<float> input_tensor;
    cv::Mat img;
    float r, dw, dh;
    if (!preprocess(image_path, input_tensor, img, r, dw, dh)) return {};

    std::array<int64_t, 4> input_shape = { 1, 3, input_image_size_.height, input_image_size_.width };
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session_.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session_.GetOutputNameAllocated(0, allocator);
    const char* input_name = input_name_alloc.get();
    const char* output_name = output_name_alloc.get();
    std::vector<const char*> input_names = { input_name };
    std::vector<const char*> output_names = { output_name };

    auto output_tensors = session_.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor_ort, 1, output_names.data(), 1);
    float* output_data = output_tensors[0].GetTensorMutableData<float>();


    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = type_info.GetShape();
    std::cout << "output_shape: ";
    for (auto v : output_shape) std::cout << v << " ";
    std::cout << std::endl;

    int num_boxes = 0, num_channels = 0;
    if (output_shape.size() == 3) { // [1, 5376, 8]
        num_boxes = static_cast<int>(output_shape[1]);
        num_channels = static_cast<int>(output_shape[2]);
    }
    else if (output_shape.size() == 2) { // [5376, 8]
        num_boxes = static_cast<int>(output_shape[0]);
        num_channels = static_cast<int>(output_shape[1]);
    }
    else {
        std::cerr << "Unexpected output shape!" << std::endl;
        return {};
    }
    std::vector<float> output(output_data, output_data + num_boxes * num_channels);
    //std::cout << "output.size() = " << output.size() << ", num_boxes = " << num_boxes << ", num_channels = " << num_channels << std::endl;
    // 需要将output, num_boxes, num_channels, img, r, dw, dh传递给postprocess
    return postprocess(output, num_boxes, num_channels, img, r, dw, dh);
}

std::vector<DetectionResult> YOLO12Infer::predict(const std::string& image_path, bool visual, bool show_score, bool show_class, bool save_or_not) {
    std::vector<DetectionResult> results = infer(image_path);
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& res = results[i];
        std::cout << "DetectionResult[" << i << "]: "
            << "class_name=" << res.class_name
            << ", bbox=[" << res.bbox.x << "," << res.bbox.y << "," << res.bbox.width << "," << res.bbox.height << "]"
            << ", confidence=" << res.confidence
            << ", area=" << res.area
            << ", area_contour=" << res.area_contour
            << ", contours=" << res.contours.size()
            << std::endl;
    }
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) return results;
    for (const auto& res : results) {
        draw_box(img, res, show_score, show_class);
    }
    if (save_or_not) {
        fs::path p(image_path);
        std::string save_dir = p.parent_path().string() + "/detection_result";
        fs::create_directories(save_dir);
        std::string save_file = save_dir + "/" + p.filename().string();
        cv::imwrite(save_file, img);
    }
    if (visual) {
        std::string img_name = fs::path(image_path).filename().string();
        std::string window_name = "Detection_" + img_name;
        cv::imshow(window_name, img);
        cv::waitKey(0);
        cv::destroyWindow(window_name);
    }
    return results;
}

void YOLO12Infer::draw_box(cv::Mat& img, const DetectionResult& res, bool show_score, bool show_class) {
    auto it = CLASS_COLORS.find(res.class_name);
    cv::Scalar color = (it != CLASS_COLORS.end()) ? it->second : cv::Scalar(0, 255, 255);
    cv::rectangle(img, res.bbox, color, 1);

    std::string label;
    if (show_class) label += res.class_name;
    if (show_score) label += (label.empty() ? "" : " ") + std::to_string(res.confidence).substr(0, 4);
    if (!label.empty()) {
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(img, cv::Rect(res.bbox.x, res.bbox.y - label_size.height - 4, label_size.width, label_size.height + 4), color, -1);
        cv::putText(img, label, cv::Point(res.bbox.x, res.bbox.y - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    // 画轮廓（黄色）
    if (!res.contours.empty()) {
        for (const auto& cnt : res.contours) {
            std::vector<cv::Point> cnt_shifted;
            for (const auto& pt : cnt) {
                cnt_shifted.emplace_back(pt.x + res.bbox.x, pt.y + res.bbox.y);
            }
            cv::drawContours(img, std::vector<std::vector<cv::Point>>{cnt_shifted}, -1, cv::Scalar(0, 255, 255), 1);
        }
    }
}