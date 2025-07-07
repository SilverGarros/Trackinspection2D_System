// Inception_TRT_DLL.cpp : 定义 DLL 的导出函数。
//

#include "pch.h"
#include "framework.h"
#include <chrono>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cfloat>  // 为了使用FLT_MAX

#include "InceptionUtils.h"
#include "InceptionTRTDLL.h"
bool TestModel_Flag = false;

namespace fs = std::filesystem;
using namespace InceptionUtils;

namespace Inception_TRT_DLL {
    const std::unordered_map<int, std::string> classes_lable_map = {
        {0, "YC"},
        {1, "DK"},
        {2, "BM"},
        {3, "HF"},
        {4, "CS"},
        {5, "ZC"},
        {6, "GF"},
        {7, "GD"}
    };
    const std::vector<std::string> CLASS_NAMES = {
        "DK_A", "DK_B", "DK_C", "CS_A", "CS_B", "CS_C","WZ","HF","GF"
    };
    const std::map<std::string, cv::Scalar> CLASS_COLORS = {
        {"DK_A", cv::Scalar(255, 0, 0)},
        {"DK_B", cv::Scalar(0, 255, 0)},
        {"DK_C", cv::Scalar(0, 0, 255)},
        {"CS_A", cv::Scalar(255, 0, 0)},
        {"CS_B", cv::Scalar(0, 255, 0)},
        {"CS_C", cv::Scalar(0, 0, 255)},
        {"WZ",   cv::Scalar(125, 0, 0)},
        {"HF",   cv::Scalar(125, 100, 0)},
        {"GF",   cv::Scalar(100, 100, 100)},
    };
}
// 添加线程本地存储缓存，减少内存分配
struct ThreadLocalCache {
    std::vector<float> input_data_cache;
    std::vector<float> output_cache;
    cv::Mat resized_img_cache;
    cv::Mat float_img_cache;
    std::vector<cv::Mat> channels_cache;

    ThreadLocalCache() {
        // 预分配常用尺寸
        input_data_cache.reserve(640 * 640 * 3);  // 预分配最大可能尺寸
        output_cache.reserve(1000);  // 预分配输出缓存
        channels_cache.resize(3);
    }
};

thread_local ThreadLocalCache tl_cache;
const std::vector<std::string> CLASS_NAMES = Inception_TRT_DLL::CLASS_NAMES;
const std::map<std::string, cv::Scalar> CLASS_COLORS = Inception_TRT_DLL::CLASS_COLORS;

namespace Inception_TRT_DLL {
    // 全局窗口管理器
    struct WindowManager {
        static std::string current_window_name;
        static bool window_created;
        static std::mutex window_mutex;

        static void ensureWindow(const std::string& window_name = "YOLO12_TensorRT_Detection") {
            std::lock_guard<std::mutex> lock(window_mutex);
            if (!window_created || current_window_name != window_name) {
                if (window_created && current_window_name != window_name) {
                    cv::destroyWindow(current_window_name);
                }
                cv::namedWindow(window_name, cv::WINDOW_NORMAL);
                current_window_name = window_name;
                window_created = true;
            }
        }

        static void showImage(const cv::Mat& img, const std::string& window_name = "YOLO12_TensorRT_Detection", int delay_ms = 1) {
            std::lock_guard<std::mutex> lock(window_mutex);
            ensureWindow(window_name);
            cv::imshow(window_name, img);
            cv::waitKey(delay_ms);  // 短暂等待，允许界面刷新
        }

        static void destroyWindow() {
            std::lock_guard<std::mutex> lock(window_mutex);
            if (window_created) {
                cv::destroyWindow(current_window_name);
                window_created = false;
                current_window_name.clear();
            }
        }
    };

    // 静态成员定义
    std::string WindowManager::current_window_name = "";
    bool WindowManager::window_created = false;
    std::mutex WindowManager::window_mutex;
}
INCEPTIONSERVICEDLL_API std::string detection_results_to_string(const std::vector<DetectionResult>& results) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& res : results) {
        nlohmann::json obj;
        obj["class_name"] = res.class_name;
        obj["bbox"] = { res.bbox.x, res.bbox.y, res.bbox.width, res.bbox.height };
        obj["confidence"] = res.confidence;
        obj["area"] = res.area;
        if (!res.contours.empty()) {
            nlohmann::json contours_json = nlohmann::json::array();
            for (const auto& contour : res.contours) {
                nlohmann::json contour_json = nlohmann::json::array();
                for (const auto& pt : contour) {
                    contour_json.push_back({ pt.x, pt.y });
                }
                contours_json.push_back(contour_json);
            }
            obj["contours"] = contours_json;
        }
        obj["area_contour"] = res.area_contour;
        arr.push_back(obj);
    }
    return arr.dump();
}


namespace Inception_TRT_DLL {
    INCEPTIONSERVICEDLL_API cv::Mat RailheadCropHighlightCenterArea(
        const cv::Mat& img, int threshold, int kernel_size, int crop_wide, bool center_limit, int limit_area)
    {
        cv::Mat img_gray;
        if (img.channels() == 3)
            cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        else
            img_gray = img.clone();

        cv::Mat binary;
        cv::threshold(img_gray, binary, threshold, 255, cv::THRESH_BINARY);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
        cv::Mat closed;
        cv::morphologyEx(binary, closed, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        int img_center = img.cols / 2;
        int crod_m = img_center;
        if (center_limit) {
            if (!contours.empty()) {
                auto largest = std::max_element(contours.begin(), contours.end(),
                    [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                        return cv::contourArea(a) < cv::contourArea(b);
                    });
                cv::Rect bbox = cv::boundingRect(*largest);
                crod_m = bbox.x + bbox.width / 2;
                if (std::abs(crod_m - img_center) > limit_area)
                    crod_m = img_center;
            }
        }

        int x1 = std::max(0, crod_m - crop_wide / 2);
        int x2 = std::min(img.cols, crod_m + crop_wide / 2);
        int y1 = 0, y2 = img.rows;
        if (x2 <= x1 || y2 <= y1) {
            x1 = std::max(0, img_center - crop_wide / 2);
            x2 = std::min(img.cols, img_center + crop_wide / 2);
        }
        return img(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    }

    INCEPTIONSERVICEDLL_API cv::Mat CropRailhead(
        const std::string& img_path, int crop_threshold, int crop_kernel_size, int crop_wide, bool center_limit, int limit_area)
    {
        cv::Mat img = imread_unicode(img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "CropRailhead: 图像读取失败: " << img_path << std::endl;
            return cv::Mat();
        }
        return RailheadCropHighlightCenterArea(img, crop_threshold, crop_kernel_size, crop_wide, center_limit, limit_area);
    }

    INCEPTIONSERVICEDLL_API std::vector<cv::Mat> StretchAndSplit(
        const cv::Mat& cropped,
        const std::string& cropped_name,
        const bool& output_or_not,
        const std::string& stretch_output_path,
        int stretch_ratio)
    {
        int orig_h = cropped.rows, orig_w = cropped.cols;
        int new_h = orig_h * stretch_ratio;
        cv::Mat stretched;
        cv::resize(cropped, stretched, cv::Size(orig_w, new_h), 0, 0, cv::INTER_LINEAR);

        int count = new_h / orig_h;
        int rem = new_h % orig_h;
        std::string base = fs::path(cropped_name).stem().string();
        std::string ext = fs::path(cropped_name).extension().string();
        std::vector<cv::Mat> stretch_piece;
        if (output_or_not) fs::create_directories(stretch_output_path);
        for (int i = 0; i < count; ++i) {
            cv::Mat piece = stretched.rowRange(i * orig_h, (i + 1) * orig_h);
            std::string out_name = base + "_" + std::to_string(count + (rem ? 1 : 0)) + "of" + std::to_string(i + 1) + ext;
            std::string out_path = stretch_output_path + "/" + out_name;
            if (output_or_not) imwrite_unicode(out_path, piece);
            stretch_piece.push_back(piece);
        }
        if (rem) {
            cv::Mat piece = stretched.rowRange(count * orig_h, new_h);
            std::string out_name = base + "_" + std::to_string(count + 1) + "of" + std::to_string(count + 1) + ext;
            std::string out_path = stretch_output_path + "/" + out_name;
            if (output_or_not) imwrite_unicode(out_path, piece);
            stretch_piece.push_back(piece);
        }
        return stretch_piece;
    }
    INCEPTIONSERVICEDLL_API std::vector<std::string> StretchAndSplit_Paths(
        const cv::Mat& cropped,
        const std::string& cropped_name,
        const bool& output_or_not,
        const std::string& stretch_output_path,
        int stretch_ratio)
    {
        int orig_h = cropped.rows, orig_w = cropped.cols;
        int new_h = orig_h * stretch_ratio;
        cv::Mat stretched;
        cv::resize(cropped, stretched, cv::Size(orig_w, new_h), 0, 0, cv::INTER_LINEAR);

        int count = new_h / orig_h;
        int rem = new_h % orig_h;
        std::string base = fs::path(cropped_name).stem().string();
        std::string ext = fs::path(cropped_name).extension().string();
        std::vector<std::string> stretch_piece_path;
        if (output_or_not) fs::create_directories(stretch_output_path);
        for (int i = 0; i < count; ++i) {
            cv::Mat piece = stretched.rowRange(i * orig_h, (i + 1) * orig_h);
            std::string out_name = base + "_" + std::to_string(count + (rem ? 1 : 0)) + "of" + std::to_string(i + 1) + ext;
            std::string out_path = stretch_output_path + "/" + out_name;
            if (output_or_not) imwrite_unicode(out_path, piece);
            stretch_piece_path.push_back(out_path);
        }
        if (rem) {
            cv::Mat piece = stretched.rowRange(count * orig_h, new_h);
            std::string out_name = base + "_" + std::to_string(count + 1) + "of" + std::to_string(count + 1) + ext;
            std::string out_path = stretch_output_path + "/" + out_name;
            if (output_or_not) imwrite_unicode(out_path, piece);
            stretch_piece_path.push_back(out_path);
        }
        return stretch_piece_path;
    }
    INCEPTIONSERVICEDLL_API std::string ClassifierTRT(const cv::Mat& img_input

    ) {
        return "DK";
    }

    INCEPTIONSERVICEDLL_API std::string Detectier_TRT(
        YOLO12TRTInfer& detector, const cv::Mat& img_input)
    {
        std::string temp_path = "temp_detection.jpg";
        imwrite_unicode(temp_path, img_input);
        return detector.predict(temp_path, false, true, true, true);
    }

    INCEPTIONSERVICEDLL_API std::string DetectImage(
        YOLO12TRTInfer& detector,
        const std::string& img_path)
    {
        try {
            cv::Mat img = imread_unicode(img_path);
            if (img.empty()) {
                std::cout << "DetectImage 获取的图像为空: " << img_path << std::endl;
                return "110 known";
            }
            if (TestModel_Flag) {
                std::cout << "DetectImage 获取到图像: " << img_path << std::endl;
            }

            auto det_results = detector.predict(img_path, false, true, true, true);
            if (!det_results.empty() && det_results.size() == 1) {
                return "ZC";
            }
            else if (det_results.empty()) {
                return "ZC";
            }
            else return det_results;
        }
        catch (const std::exception& e) {
            std::cerr << "DetectImage 异常: " << e.what() << std::endl;
            return "DetectImage Exception";
        }
        catch (...) {
            std::cerr << "DetectImage 未知异常" << std::endl;
            return "DetectImage Unknown Exception";
        }
    }

} // namespace Inception_TRT_DLL

// TensorRT Classifier Implementation
INCEPTIONSERVICEDLL_API Classifier_TRT_Infer::Classifier_TRT_Infer(const std::string& engine_file, cv::Size input_image_size)
    : input_image_size_(input_image_size), input_device_buffer_(nullptr), output_device_buffer_(nullptr), stream_(nullptr) {
    if (!loadEngine(engine_file)) {
        throw std::runtime_error("Failed to load engine.");
    }
}
Classifier_TRT_Infer::~Classifier_TRT_Infer() {
    // 添加适当的资源清理
    if (input_device_buffer_) {
        cudaFree(input_device_buffer_);
        input_device_buffer_ = nullptr;
    }
    if (output_device_buffer_) {
        cudaFree(output_device_buffer_);
        output_device_buffer_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}
bool Classifier_TRT_Infer::loadEngine(const std::string& engine_file) {
    std::ifstream file(engine_file, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error loading engine file: " << engine_file << std::endl;
        return false;
    }
    file.seekg(0, file.end);
    size_t engine_size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engine_data(engine_size);
    file.read(engine_data.data(), engine_size);
    file.close();

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), engine_size));
    if (!engine_) return false;

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) return false;

    int32_t nb_io_tensors = engine_->getNbIOTensors();
    for (int32_t i = 0; i < nb_io_tensors; ++i) {
        const char* tensor_name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);
        if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
            input_tensor_name_ = std::string(tensor_name);
        }
        else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_tensor_name_ = std::string(tensor_name);
        }
    }

    // 获取张量形状信息
    auto input_dims = engine_->getTensorShape(input_tensor_name_.c_str());
    auto output_dims = engine_->getTensorShape(output_tensor_name_.c_str());

    input_size_ = 1;
    for (int i = 0; i < input_dims.nbDims; ++i) {
        input_size_ *= input_dims.d[i];
    }
    output_size_ = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size_ *= output_dims.d[i];
    }

    // 分配显存
    cudaMalloc(&input_device_buffer_, input_size_ * sizeof(float));
    cudaMalloc(&output_device_buffer_, output_size_ * sizeof(float));
    cudaStreamCreate(&stream_);

    return true;
}

INCEPTIONSERVICEDLL_API ClassificationResult Classifier_TRT_Infer::predict(const std::string& image_path)
{
    auto& input_data = tl_cache.input_data_cache;
    auto& resized_img = tl_cache.resized_img_cache;
    auto& float_img = tl_cache.float_img_cache;
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) throw std::runtime_error("Failed to read image: " + image_path);

    cv::resize(img, img, input_image_size_);
    img.convertTo(img, CV_32F, 1.0 / 255);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    input_data.resize(input_size_);
    int index = 0;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < img.rows; ++h) {
            for (int w = 0; w < img.cols; ++w) {
                input_data[index++] = img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    cudaMemcpyAsync(input_device_buffer_, input_data.data(), input_size_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->setInputTensorAddress(input_tensor_name_.c_str(), input_device_buffer_);
    context_->setOutputTensorAddress(output_tensor_name_.c_str(), output_device_buffer_);

    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT execution failed.");
    }

    auto& output = tl_cache.output_cache;
    output.resize(output_size_);
    cudaMemcpyAsync(output.data(), output_device_buffer_,
        output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // 应用 softmax 函数获得概率分布
    std::vector<float> softmax_output(output.size());
    float max_val = *std::max_element(output.begin(), output.end());
    float sum_exp = 0.0f;

    // 计算 exp(x - max) 避免数值溢出
    for (size_t i = 0; i < output.size(); ++i) {
        softmax_output[i] = std::exp(output[i] - max_val);
        sum_exp += softmax_output[i];
    }
    // 归一化
    for (size_t i = 0; i < softmax_output.size(); ++i) {
        softmax_output[i] /= sum_exp;
    }

    int max_index = std::distance(softmax_output.begin(), std::max_element(softmax_output.begin(), softmax_output.end()));
    float confidence = softmax_output[max_index];

    ClassificationResult result;
    result.class_id = max_index;
    result.confidence = confidence;
    //if (max_index < static_cast<int>(Inception_TRT_DLL::CLASS_NAMES.size())) {
    //    result.class_name = Inception_TRT_DLL::CLASS_NAMES[max_index];
    //}
    //else {
    //    result.class_name = "Unknown";
    //}
    // 使用 classes_lable_map 进行映射
    auto it = Inception_TRT_DLL::classes_lable_map.find(max_index);
    if (it != Inception_TRT_DLL::classes_lable_map.end()) {
        result.class_name = it->second;
    }
    else {
        result.class_name = "Unknown";
    }
    return result;

}
INCEPTIONSERVICEDLL_API ClassificationResult Classifier_TRT_Infer::predict(cv::Mat image) {
    auto& input_data = tl_cache.input_data_cache;
    auto& resized_img = tl_cache.resized_img_cache;
    auto& float_img = tl_cache.float_img_cache;

    if (image.empty()) throw std::runtime_error("Failed to read image");
    cv::Mat img;
    cv::resize(image, img, input_image_size_);
    img.convertTo(img, CV_32F, 1.0 / 255);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    input_data.resize(input_size_);
    int index = 0;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < img.rows; ++h) {
            for (int w = 0; w < img.cols; ++w) {
                input_data[index++] = img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    cudaMemcpyAsync(input_device_buffer_, input_data.data(), input_size_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->setInputTensorAddress(input_tensor_name_.c_str(), input_device_buffer_);
    context_->setOutputTensorAddress(output_tensor_name_.c_str(), output_device_buffer_);

    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT execution failed.");
    }

    auto& output = tl_cache.output_cache;
    output.resize(output_size_);
    cudaMemcpyAsync(output.data(), output_device_buffer_,
        output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // 应用 softmax 函数获得概率分布
    std::vector<float> softmax_output(output.size());
    float max_val = *std::max_element(output.begin(), output.end());
    float sum_exp = 0.0f;

    // 计算 exp(x - max) 避免数值溢出
    for (size_t i = 0; i < output.size(); ++i) {
        softmax_output[i] = std::exp(output[i] - max_val);
        sum_exp += softmax_output[i];
    }

    // 归一化
    for (size_t i = 0; i < softmax_output.size(); ++i) {
        softmax_output[i] /= sum_exp;
    }

    int max_index = std::distance(softmax_output.begin(), std::max_element(softmax_output.begin(), softmax_output.end()));
    float confidence = softmax_output[max_index];  // 使用 softmax 后的概率作为置信度

    ClassificationResult result;
    result.class_id = max_index;
    result.confidence = confidence;
    //if (max_index < static_cast<int>(Inception_TRT_DLL::CLASS_NAMES.size())) {
    //    result.class_name = Inception_TRT_DLL::CLASS_NAMES[max_index];
    //}
    //else {
    //    result.class_name = "Unknown";
    //}
    //使用 classes_lable_map 进行映射
    auto it = Inception_TRT_DLL::classes_lable_map.find(max_index);
    if (it != Inception_TRT_DLL::classes_lable_map.end()) {
        result.class_name = it->second;
    }
    else {
        result.class_name = "Unknown";
    }
    return result;
}

// TensorRT YOLO12 Implementation
YOLO12TRTInfer::YOLO12TRTInfer(const std::string& engine_file,
    cv::Size input_image_size,
    float confidence_thres,
    float iou_thres)
    : input_image_size_(input_image_size),
    confidence_thres_(confidence_thres),
    iou_thres_(iou_thres),
    input_device_buffer_(nullptr),
    output_device_buffer_(nullptr),
    stream_(nullptr),
    input_binding_index_(-1),
    output_binding_index_(-1)
{
    // 🔧 初始化多流数组为nullptr
    for (int i = 0; i < NUM_STREAMS; ++i) {
        streams_[i] = nullptr;
        input_buffers_[i] = nullptr;
        output_buffers_[i] = nullptr;
    }

    // 🔧 先加载引擎，获取正确的缓冲区大小
    if (!loadEngine(engine_file)) {
        throw std::runtime_error("Failed to load TensorRT engine: " + engine_file);
    }

    // 🔧 获取输入输出尺寸
    input_width_ = input_image_size_.width;
    input_height_ = input_image_size_.height;

    // 🔧 计算缓冲区大小
    input_size_ = 1 * 3 * input_height_ * input_width_ * sizeof(float);

    nvinfer1::Dims output_dims = engine_->getTensorShape(output_tensor_name_.c_str());
    size_t output_elements = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_elements *= static_cast<size_t>(output_dims.d[i]);
    }
    output_size_ = output_elements * sizeof(float);

    // 🔧 现在可以安全地初始化多个CUDA流
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaError_t cuda_status = cudaStreamCreate(&streams_[i]);
        if (cuda_status != cudaSuccess) {
            // 清理已创建的流
            for (int j = 0; j < i; ++j) {
                if (streams_[j]) cudaStreamDestroy(streams_[j]);
            }
            throw std::runtime_error("Failed to create CUDA stream " + std::to_string(i) +
                ": " + std::string(cudaGetErrorString(cuda_status)));
        }

        // 🔧 为每个流分配GPU缓冲区
        cuda_status = cudaMalloc(&input_buffers_[i], input_size_);
        if (cuda_status != cudaSuccess) {
            // 清理已分配的资源
            for (int j = 0; j <= i; ++j) {
                if (streams_[j]) cudaStreamDestroy(streams_[j]);
                if (j < i && input_buffers_[j]) cudaFree(input_buffers_[j]);
            }
            throw std::runtime_error("Failed to allocate input buffer for stream " + std::to_string(i) +
                ": " + std::string(cudaGetErrorString(cuda_status)));
        }

        cuda_status = cudaMalloc(&output_buffers_[i], output_size_);
        if (cuda_status != cudaSuccess) {
            // 清理已分配的资源
            for (int j = 0; j <= i; ++j) {
                if (streams_[j]) cudaStreamDestroy(streams_[j]);
                if (input_buffers_[j]) cudaFree(input_buffers_[j]);
                if (j < i && output_buffers_[j]) cudaFree(output_buffers_[j]);
            }
            throw std::runtime_error("Failed to allocate output buffer for stream " + std::to_string(i) +
                ": " + std::string(cudaGetErrorString(cuda_status)));
        }
    }

    // 🔧 保留原有的单流支持（向后兼容）
    stream_ = streams_[0];
    input_device_buffer_ = input_buffers_[0];
    output_device_buffer_ = output_buffers_[0];
}

YOLO12TRTInfer::~YOLO12TRTInfer() {
    // 安全清理多流资源
    for (int i = 0; i < NUM_STREAMS; ++i) {
        if (streams_[i]) {
            cudaStreamSynchronize(streams_[i]); // 确保流完成
            cudaStreamDestroy(streams_[i]);
            streams_[i] = nullptr;
        }
        if (input_buffers_[i]) {
            cudaFree(input_buffers_[i]);
            input_buffers_[i] = nullptr;
        }
        if (output_buffers_[i]) {
            cudaFree(output_buffers_[i]);
            output_buffers_[i] = nullptr;
        }
    }

    // 清理原有资源（如果不是多流中的一部分）
    if (stream_ && stream_ != streams_[0]) {
        cudaStreamDestroy(stream_);
    }
    stream_ = nullptr;
    input_device_buffer_ = nullptr;
    output_device_buffer_ = nullptr;
}
int YOLO12TRTInfer::getAvailableStream() {
    std::lock_guard<std::mutex> lock(stream_mutex_);

    // 轮询分配流
    int stream_id = current_stream_id_.fetch_add(1) % NUM_STREAMS;

    // 🔧 确保流已完成之前的操作
    cudaError_t status = cudaStreamSynchronize(streams_[stream_id]);
    if (status != cudaSuccess) {
        std::cerr << "Warning: Stream synchronization failed for stream " << stream_id
            << ": " << cudaGetErrorString(status) << std::endl;
    }

    return stream_id;
}

bool YOLO12TRTInfer::loadEngine(const std::string& engine_file) {
    std::ifstream file(engine_file, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to read engine file: " << engine_file << std::endl;
        return false;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
        std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create TensorRT execution context" << std::endl;
        return false;
    }

    int32_t nb_io_tensors = engine_->getNbIOTensors();
    for (int32_t i = 0; i < nb_io_tensors; ++i) {
        const char* tensor_name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);

        if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
            input_tensor_name_ = std::string(tensor_name);
        }
        else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_tensor_name_ = std::string(tensor_name);
        }
    }

    if (input_tensor_name_.empty() || output_tensor_name_.empty()) {
        std::cerr << "Failed to find input/output tensors" << std::endl;
        return false;
    }

    return true;
}

INCEPTIONSERVICEDLL_API cv::Mat YOLO12TRTInfer::letterbox(const cv::Mat& img, float& h_ratio, float& w_ratio) {
    cv::Size target_size = input_image_size_;
    cv::Mat resized;
    cv::resize(img, resized, target_size, 0, 0, cv::INTER_LINEAR);
    h_ratio = static_cast<float>(target_size.height) / img.rows;
    w_ratio = static_cast<float>(target_size.width) / img.cols;
    return resized;
}

INCEPTIONSERVICEDLL_API std::vector<float> YOLO12TRTInfer::preprocess(const std::string& image_path, cv::Mat& original_img, float& h_ratio, float& w_ratio) {
    original_img = InceptionUtils::imread_unicode(image_path, cv::IMREAD_COLOR);
    if (original_img.empty()) throw std::runtime_error("无法读取图像: " + image_path);

    if (TestModel_Flag == true) {
        std::cout << "C++ TensorRT 原始图像信息: " << image_path << std::endl;
        std::cout << "原始图像尺寸: " << original_img.cols << "x" << original_img.rows << std::endl;
    }

    // 从BGR转为RGB
    cv::Mat img_rgb;
    cv::cvtColor(original_img, img_rgb, cv::COLOR_BGR2RGB);

    // 调整大小
    cv::Mat img_resized = letterbox(img_rgb, h_ratio, w_ratio);

    if (TestModel_Flag == true) {
        std::cout << "Letterbox后尺寸: " << img_resized.cols << "x" << img_resized.rows << std::endl;
        std::cout << "缩放比例: w_ratio=" << w_ratio << ", h_ratio=" << h_ratio << std::endl;
    }

    // 归一化
    cv::Mat img_float;
    img_resized.convertTo(img_float, CV_32F, 1.0 / 255.0);

    // 将图像转换为CHW格式
    std::vector<cv::Mat> chw(3);
    cv::split(img_float, chw);
    std::vector<float> input_tensor;
    input_tensor.reserve(img_resized.rows * img_resized.cols * 3);

    // 按照R,G,B顺序添加通道数据
    for (int c = 0; c < 3; ++c) {
        const float* channel_data = chw[c].ptr<float>();
        input_tensor.insert(input_tensor.end(), channel_data, channel_data + img_resized.rows * img_resized.cols);
    }

    if (TestModel_Flag == true) {
        size_t tensor_size = input_tensor.size();
        float sum = 0.0f, min_val = FLT_MAX, max_val = -FLT_MAX;
        for (size_t i = 0; i < std::min(tensor_size, static_cast<size_t>(100)); ++i) {
            sum += input_tensor[i];
            min_val = std::min(min_val, input_tensor[i]);
            max_val = std::max(max_val, input_tensor[i]);
        }

        std::cout << "C++ TensorRT 输入tensor统计 (前100个元素):" << std::endl;
        std::cout << "- 元素总数: " << tensor_size << std::endl;
        std::cout << "- 形状: [1, 3, " << img_resized.rows << ", " << img_resized.cols << "]" << std::endl;
        std::cout << "- 均值: " << (sum / std::min(tensor_size, static_cast<size_t>(100))) << std::endl;
        std::cout << "- 最小值: " << min_val << std::endl;
        std::cout << "- 最大值: " << max_val << std::endl;
    }
    return input_tensor;
}

INCEPTIONSERVICEDLL_API std::vector<float> YOLO12TRTInfer::preprocess(const cv::Mat img, cv::Mat& original_img, float& h_ratio, float& w_ratio) {
    original_img = img;
    if (original_img.empty()) throw std::runtime_error("preprocess无法读取图像: ");

    if (TestModel_Flag == true) {
        std::cout << "原始图像尺寸: " << original_img.cols << "x" << original_img.rows << std::endl;
    }

    // 从BGR转为RGB
    cv::Mat img_rgb;
    cv::cvtColor(original_img, img_rgb, cv::COLOR_BGR2RGB);

    // 调整大小
    cv::Mat img_resized = letterbox(img_rgb, h_ratio, w_ratio);

    if (TestModel_Flag == true) {
        std::cout << "Letterbox后尺寸: " << img_resized.cols << "x" << img_resized.rows << std::endl;
        std::cout << "缩放比例: w_ratio=" << w_ratio << ", h_ratio=" << h_ratio << std::endl;
    }

    // 归一化
    cv::Mat img_float;
    img_resized.convertTo(img_float, CV_32F, 1.0 / 255.0);

    // 将图像转换为CHW格式
    std::vector<cv::Mat> chw(3);
    cv::split(img_float, chw);
    std::vector<float> input_tensor;
    input_tensor.reserve(img_resized.rows * img_resized.cols * 3);

    // 按照R,G,B顺序添加通道数据
    for (int c = 0; c < 3; ++c) {
        const float* channel_data = chw[c].ptr<float>();
        input_tensor.insert(input_tensor.end(), channel_data, channel_data + img_resized.rows * img_resized.cols);
    }

    if (TestModel_Flag == true) {
        size_t tensor_size = input_tensor.size();
        float sum = 0.0f, min_val = FLT_MAX, max_val = -FLT_MAX;
        for (size_t i = 0; i < std::min(tensor_size, static_cast<size_t>(100)); ++i) {
            sum += input_tensor[i];
            min_val = std::min(min_val, input_tensor[i]);
            max_val = std::max(max_val, input_tensor[i]);
        }

        std::cout << "C++ TensorRT 输入tensor统计 (前100个元素):" << std::endl;
        std::cout << "- 元素总数: " << tensor_size << std::endl;
        std::cout << "- 形状: [1, 3, " << img_resized.rows << ", " << img_resized.cols << "]" << std::endl;
        std::cout << "- 均值: " << (sum / std::min(tensor_size, static_cast<size_t>(100))) << std::endl;
        std::cout << "- 最小值: " << min_val << std::endl;
        std::cout << "- 最大值: " << max_val << std::endl;
    }
    return input_tensor;
}
INCEPTIONSERVICEDLL_API std::vector<DetectionResult> YOLO12TRTInfer::postprocess(const std::vector<float>& output, int rows, int cols,
    float h_ratio, float w_ratio, const cv::Mat& original_img) {
    std::vector<std::vector<float>> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<DetectionResult> results;

    // 遍历每一行，每一行代表一个检测框
    for (int i = 0; i < rows; ++i) {
        const float* row = &output[i * cols];
        float max_class_score = 0.0f;
        int max_class_idx = -1;

        // 从索引5开始遍历，查找最大类别得分
        for (int j = 5; j < std::min(5 + static_cast<int>(Inception_TRT_DLL::CLASS_NAMES.size()), cols); ++j) {
            
            float sigmoid_score = 1.0f / (1.0f + std::exp(-row[j]));

            if (sigmoid_score > max_class_score) {
                max_class_score = sigmoid_score;
                max_class_idx = j - 5;
            }
        }

        float confidence = max_class_score;

        if (confidence >= confidence_thres_ && max_class_idx >= 0) {
            float x_center = row[0];
            float y_center = row[1];
            float width = row[2];
            float height = row[3];

            // 将坐标从网络输入尺寸映射回原始图像尺寸
            float orig_x_center = x_center / w_ratio;
            float orig_y_center = y_center / h_ratio;
            float orig_width = width / w_ratio;
            float orig_height = height / h_ratio;

            // 计算原始图像上的左上角坐标和宽高
            int left = std::max(0, static_cast<int>(orig_x_center - orig_width / 2));
            int top = std::max(0, static_cast<int>(orig_y_center - orig_height / 2));
            int box_width = std::min(static_cast<int>(orig_width), original_img.cols - left);
            int box_height = std::min(static_cast<int>(orig_height), original_img.rows - top);

            boxes.push_back({ static_cast<float>(left), static_cast<float>(top),
                              static_cast<float>(box_width), static_cast<float>(box_height) });
            scores.push_back(confidence);
            class_ids.push_back(max_class_idx);
        }
    }

    // NMS
    std::vector<cv::Rect> cv_boxes;
    for (const auto& b : boxes) {
        cv_boxes.emplace_back(static_cast<int>(b[0]), static_cast<int>(b[1]), static_cast<int>(b[2]), static_cast<int>(b[3]));
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(cv_boxes, scores, confidence_thres_, iou_thres_, indices);

    // 创建结果对象
    for (int idx : indices) {
        DetectionResult res;
        int class_id = class_ids[idx];

        if (class_id >= 0 && class_id < static_cast<int>(Inception_TRT_DLL::CLASS_NAMES.size())) {
            res.class_name = Inception_TRT_DLL::CLASS_NAMES[class_id];
        }
        else {
            std::cout << "警告：类别ID(" << class_id << ")超出范围" << std::endl;
            res.class_name = "UnknownID_" + std::to_string(class_id);
        }
        res.bbox = cv_boxes[idx];
        res.confidence = scores[idx];
        res.area = res.bbox.width * res.bbox.height;

        // DK类处理（与原代码相同的后处理逻辑）
        if (res.class_name.find("DK") != std::string::npos) {
            if (TestModel_Flag) {
                std::cout << "执行掉块类处理....." << std::endl;
            }

            int x1 = std::max(res.bbox.x, 0);
            int y1 = std::max(res.bbox.y, 0);
            int x2 = std::min(res.bbox.x + res.bbox.width, original_img.cols);
            int y2 = std::min(res.bbox.y + res.bbox.height, original_img.rows);
            cv::Mat crop = original_img(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();

            cv::Mat gray, gray_inv, binary;
            if (crop.channels() == 3)
                cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
            else
                gray = crop;

            cv::Mat lut(1, 256, CV_8U);
            uchar* p = lut.ptr();
            float inv_gamma = 1.0f / 2.0f;
            for (int i = 0; i < 256; ++i) p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, inv_gamma) * 255.0);
            cv::LUT(gray, lut, gray);

            cv::bitwise_not(gray, gray_inv);
            cv::adaptiveThreshold(gray_inv, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, -10);
            cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            double area_contour = 0.0;
            for (const auto& cnt : contours) {
                double a = cv::contourArea(cnt);
                if (a > 15) area_contour += a;
            }

            if (area_contour > res.area) area_contour = res.area;
            res.contours = contours;
            res.area_contour = static_cast<float>(area_contour);
        }

        results.push_back(res);
    }
    return results;
}

INCEPTIONSERVICEDLL_API std::vector<DetectionResult> YOLO12TRTInfer::infer(const std::string& image_path) {
    cv::Mat original_img;
    float h_ratio, w_ratio;
    std::vector<float> input_tensor = preprocess(image_path, original_img, h_ratio, w_ratio);

    // Copy input data to GPU
    cudaError_t cuda_status = cudaMemcpyAsync(input_device_buffer_, input_tensor.data(), input_size_, cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy input data to GPU: " << cudaGetErrorString(cuda_status) << std::endl;
        return {};
    }

    // 设置输入tensor形状 - 使用新API
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 1;
    input_dims.d[1] = 3;
    input_dims.d[2] = input_height_;
    input_dims.d[3] = input_width_;

    context_->setInputShape(input_tensor_name_.c_str(), input_dims);

    // 设置tensor地址 - 使用新API
    context_->setTensorAddress(input_tensor_name_.c_str(), input_device_buffer_);
    context_->setTensorAddress(output_tensor_name_.c_str(), output_device_buffer_);

    // Execute inference - 使用新API
    bool status = context_->enqueueV3(stream_);
    if (!status) {
        std::cerr << "TensorRT inference failed" << std::endl;
        return {};
    }

    // Copy output data from GPU
    std::vector<float> output_data(output_size_ / sizeof(float));
    cuda_status = cudaMemcpyAsync(output_data.data(), output_device_buffer_, output_size_, cudaMemcpyDeviceToHost, stream_);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy output data from GPU: " << cudaGetErrorString(cuda_status) << std::endl;
        return {};
    }

    cudaStreamSynchronize(stream_);

    // Get output dimensions
    nvinfer1::Dims output_dims = engine_->getTensorShape(output_tensor_name_.c_str());
    int orig_rows = static_cast<int>(output_dims.d[1]);  // 通常是类别数 + 4
    int orig_cols = static_cast<int>(output_dims.d[2]);  // 检测框数量

    // 转置输出数据 [1, 10, 5376] -> [5376, 10]
    std::vector<float> transposed_output(orig_cols * orig_rows);
    for (int i = 0; i < orig_rows; ++i) {
        for (int j = 0; j < orig_cols; ++j) {
            transposed_output[j * orig_rows + i] = output_data[i * orig_cols + j];
        }
    }

    return postprocess(transposed_output, orig_cols, orig_rows, h_ratio, w_ratio, original_img);
}

INCEPTIONSERVICEDLL_API std::vector<DetectionResult> YOLO12TRTInfer::infer(const cv::Mat& img) {
    cv::Mat original_img;
    float h_ratio, w_ratio;
    std::vector<float> input_tensor = preprocess(img, original_img, h_ratio, w_ratio);

    // Copy input data to GPU
    cudaError_t cuda_status = cudaMemcpyAsync(input_device_buffer_, input_tensor.data(), input_size_, cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy input data to GPU: " << cudaGetErrorString(cuda_status) << std::endl;
        return {};
    }

    // 设置输入tensor形状 - 使用新API
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 1;
    input_dims.d[1] = 3;
    input_dims.d[2] = input_height_;
    input_dims.d[3] = input_width_;

    context_->setInputShape(input_tensor_name_.c_str(), input_dims);

    // 设置tensor地址 - 使用新API
    context_->setTensorAddress(input_tensor_name_.c_str(), input_device_buffer_);
    context_->setTensorAddress(output_tensor_name_.c_str(), output_device_buffer_);

    // Execute inference - 使用新API
    bool status = context_->enqueueV3(stream_);
    if (!status) {
        std::cerr << "TensorRT inference failed" << std::endl;
        return {};
    }

    // Copy output data from GPU
    std::vector<float> output_data(output_size_ / sizeof(float));
    cuda_status = cudaMemcpyAsync(output_data.data(), output_device_buffer_, output_size_, cudaMemcpyDeviceToHost, stream_);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy output data from GPU: " << cudaGetErrorString(cuda_status) << std::endl;
        return {};
    }

    cudaStreamSynchronize(stream_);

    // Get output dimensions
    nvinfer1::Dims output_dims = engine_->getTensorShape(output_tensor_name_.c_str());
    int orig_rows = static_cast<int>(output_dims.d[1]);  // 通常是类别数 + 4
    int orig_cols = static_cast<int>(output_dims.d[2]);  // 检测框数量

    // 转置输出数据 [1, 10, 5376] -> [5376, 10]
    std::vector<float> transposed_output(orig_cols * orig_rows);
    for (int i = 0; i < orig_rows; ++i) {
        for (int j = 0; j < orig_cols; ++j) {
            transposed_output[j * orig_rows + i] = output_data[i * orig_cols + j];
        }
    }

    return postprocess(transposed_output, orig_cols, orig_rows, h_ratio, w_ratio, original_img);
}


std::vector<DetectionResult> YOLO12TRTInfer::inferWithMultiStream(const cv::Mat& img) {
    int stream_id = getAvailableStream();
    cudaStream_t current_stream = streams_[stream_id];
    void* current_input_buffer = input_buffers_[stream_id];
    void* current_output_buffer = output_buffers_[stream_id];

    cv::Mat original_img;
    float h_ratio, w_ratio;
    std::vector<float> input_tensor = preprocess(img, original_img, h_ratio, w_ratio);

    // 🚀 异步数据传输到GPU
    cudaError_t cuda_status = cudaMemcpyAsync(
        current_input_buffer,
        input_tensor.data(),
        input_size_,
        cudaMemcpyHostToDevice,
        current_stream
    );
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy input data to GPU stream " << stream_id << std::endl;
        return {};
    }
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 1;
    input_dims.d[1] = 3;
    input_dims.d[2] = input_height_;
    input_dims.d[3] = input_width_;

    context_->setInputShape(input_tensor_name_.c_str(), input_dims);
    context_->setTensorAddress(input_tensor_name_.c_str(), current_input_buffer);
    context_->setTensorAddress(output_tensor_name_.c_str(), current_output_buffer);
    bool status = context_->enqueueV3(current_stream);
    if (!status) {
        std::cerr << "TensorRT inference failed on stream " << stream_id << std::endl;
        return {};
    }
    std::vector<float> output_data(output_size_ / sizeof(float));
    cuda_status = cudaMemcpyAsync(
        output_data.data(),
        current_output_buffer,
        output_size_,
        cudaMemcpyDeviceToHost,
        current_stream
    );
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy output data from GPU stream " << stream_id << std::endl;
        return {};
    }
    cudaStreamSynchronize(current_stream);

    // 后处理（在CPU上进行）
    nvinfer1::Dims output_dims = engine_->getTensorShape(output_tensor_name_.c_str());
    int orig_rows = static_cast<int>(output_dims.d[1]);
    int orig_cols = static_cast<int>(output_dims.d[2]);

    std::vector<float> transposed_output(orig_cols * orig_rows);
    for (int i = 0; i < orig_rows; ++i) {
        for (int j = 0; j < orig_cols; ++j) {
            transposed_output[j * orig_rows + i] = output_data[i * orig_cols + j];
        }
    }

    return postprocess(transposed_output, orig_cols, orig_rows, h_ratio, w_ratio, original_img);
}


INCEPTIONSERVICEDLL_API void YOLO12TRTInfer::draw_box(cv::Mat& img, const DetectionResult& res, bool show_score, bool show_class) {
    auto it = Inception_TRT_DLL::CLASS_COLORS.find(res.class_name);
    cv::Scalar color = (it != Inception_TRT_DLL::CLASS_COLORS.end()) ? it->second : cv::Scalar(0, 255, 255);
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

    // 画轮廓（红色）
    if (!res.contours.empty() && res.class_name.find("DK") != std::string::npos) {
        for (const auto& cnt : res.contours) {
            std::vector<cv::Point> cnt_shifted;
            for (const auto& pt : cnt) {
                cnt_shifted.emplace_back(pt.x + res.bbox.x, pt.y + res.bbox.y);
            }
            cv::drawContours(img, std::vector<std::vector<cv::Point>>{cnt_shifted}, -1, cv::Scalar(0, 0, 255), 1);
        }
    }
}

INCEPTIONSERVICEDLL_API std::string YOLO12TRTInfer::predict(const std::string& image_path, bool visual, bool show_score, bool show_class, bool save_or_not) {
    std::vector<DetectionResult> results = infer(image_path);
    cv::Mat img = InceptionUtils::imread_unicode(image_path, cv::IMREAD_COLOR);
    if (results.empty()) {
        return R"([{"class_name": "ZC"}])";
    }
    for (const auto& res : results) {
        draw_box(img, res, show_score, show_class);
    }
    if (save_or_not) {
        fs::path p(image_path);
        std::string save_dir = p.parent_path().string() + "//detection_result";
        fs::create_directories(save_dir);
        std::string save_file = save_dir + "//" + p.filename().string();
        InceptionUtils::imwrite_unicode(save_file, img);
    }
    if (visual) {
        // 使用窗口管理器自动刷新显示
        Inception_TRT_DLL::WindowManager::showImage(img, "YOLO12_TensorRT_Detection", 1);
    }
    return detection_results_to_string(results);
}

INCEPTIONSERVICEDLL_API std::vector<DetectionResult> YOLO12TRTInfer::predict(cv::Mat& img, bool visual, bool show_score, bool show_class, bool save_or_not, std::string img_path) {
    std::vector<DetectionResult> results = infer(img);
    if (results.empty()) {
        return std::vector<DetectionResult>();
    }
    for (const auto& res : results) {
        draw_box(img, res, show_score, show_class);
    }
    if (save_or_not) {
        fs::path p(img_path);
        std::string save_dir = p.parent_path().string() + "//detection_result";
        fs::create_directories(save_dir);
        std::string save_file = save_dir + "//" + p.filename().string();
        InceptionUtils::imwrite_unicode(save_file, img);
    }
    if (visual) {
        // 使用窗口管理器自动刷新显示
        Inception_TRT_DLL::WindowManager::showImage(img, "YOLO12_TensorRT_Detection", 1);
    }
    return results;
}
struct DKProcessingCache {
    cv::Mat crop_img;
    cv::Mat gray_img;
    cv::Mat gray_inv;
    cv::Mat binary_img;
    cv::Mat lut_table;
    cv::Mat morph_kernel;
    std::vector<std::vector<cv::Point>> contours_cache;
    std::vector<cv::Vec4i> hierarchy_cache;

    DKProcessingCache() {
        // 预创建LUT表，避免重复计算
        lut_table.create(1, 256, CV_8U);
        uchar* lut_ptr = lut_table.ptr<uchar>();
        const float inv_gamma = 1.0f / 2.0f;
        for (int i = 0; i < 256; ++i) {
            lut_ptr[i] = cv::saturate_cast<uchar>(pow(i / 255.0, inv_gamma) * 255.0);
        }

        // 预创建形态学核
        morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

        // 预分配轮廓容器
        contours_cache.reserve(100);
        hierarchy_cache.reserve(100);
    }
};

thread_local DKProcessingCache dk_cache;
INCEPTIONSERVICEDLL_API void process_dk_detection_optimized(DetectionResult& res, const cv::Mat& original_img) {
    // 使用线程本地缓存，避免重复分配
    auto& crop = dk_cache.crop_img;
    auto& gray = dk_cache.gray_img;
    auto& gray_inv = dk_cache.gray_inv;
    auto& binary = dk_cache.binary_img;
    auto& lut_table = dk_cache.lut_table;
    auto& kernel = dk_cache.morph_kernel;
    auto& contours = dk_cache.contours_cache;
    auto& hierarchy = dk_cache.hierarchy_cache;

    // 清理容器
    contours.clear();
    hierarchy.clear();

    // 边界检查优化
    const int x1 = std::max(res.bbox.x, 0);
    const int y1 = std::max(res.bbox.y, 0);
    const int x2 = std::min(res.bbox.x + res.bbox.width, original_img.cols);
    const int y2 = std::min(res.bbox.y + res.bbox.height, original_img.rows);

    // 验证ROI有效性
    if (x2 <= x1 || y2 <= y1) {
        res.area_contour = 0.0f;
        return;
    }

    const cv::Rect roi(x1, y1, x2 - x1, y2 - y1);

    // 优化1：直接在ROI上操作，避免clone()
    const cv::Mat crop_roi = original_img(roi);

    // 优化2：根据输入图像通道数选择最优转换路径
    if (crop_roi.channels() == 3) {
        cv::cvtColor(crop_roi, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        crop_roi.copyTo(gray);
    }

    // 优化3：使用预计算的LUT表，避免重复计算Gamma校正
    cv::LUT(gray, lut_table, gray);

    // 优化4：原地操作，减少内存分配
    cv::bitwise_not(gray, gray_inv);

    // 优化5：使用更快的阈值化参数
    cv::adaptiveThreshold(gray_inv, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, -10);

    // 优化6：使用预创建的形态学核
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

    // 优化7：使用优化的轮廓检测
    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 优化8：快速面积计算，使用阈值过滤
    double total_area = 0.0;
    constexpr double MIN_CONTOUR_AREA = 15.0;  // 编译时常量

    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area > MIN_CONTOUR_AREA) {
            total_area += area;
        }
    }

    // 优化9：边界检查
    if (total_area > res.area) {
        total_area = res.area;
    }

    // 结果赋值
    res.contours = std::move(contours);
    res.area_contour = static_cast<float>(total_area);
}
INCEPTIONSERVICEDLL_API std::vector<DetectionResult> YOLO12TRTInfer::postprocess_optimized(
    const std::vector<float>& output, int rows, int cols,
    float h_ratio, float w_ratio, const cv::Mat& original_img) {

    // 预分配容器，避免动态扩容
    std::vector<DetectionResult> results;
    results.reserve(32);

    static thread_local std::vector<std::vector<float>> boxes;
    static thread_local std::vector<float> scores;
    static thread_local std::vector<int> class_ids;
    static thread_local std::vector<cv::Rect> cv_boxes;

    boxes.clear();
    scores.clear();
    class_ids.clear();
    cv_boxes.clear();

    // 预分配空间
    const int estimated_detections = std::min(rows / 10, 100);
    boxes.reserve(estimated_detections);
    scores.reserve(estimated_detections);
    class_ids.reserve(estimated_detections);
    cv_boxes.reserve(estimated_detections);

    // 第一阶段：候选框生成 - 向量化优化
    const int class_start = 5;
    const int class_end = std::min(class_start + static_cast<int>(Inception_TRT_DLL::CLASS_NAMES.size()), cols);

    for (int i = 0; i < rows; ++i) {
        const float* row = &output[i * cols];

        // 快速查找最大类别得分 - 手动展开循环
        float max_score = row[class_start];
        int max_idx = 0;

        for (int j = class_start + 1; j < class_end; ++j) {
            if (row[j] > max_score) {
                max_score = row[j];
                max_idx = j - class_start;
            }
        }

        // 早期退出优化
        if (max_score < confidence_thres_) continue;

        // 坐标转换优化 - 减少除法运算
        const float x_center = row[0];
        const float y_center = row[1];
        const float width = row[2];
        const float height = row[3];

        const float inv_w_ratio = 1.0f / w_ratio;
        const float inv_h_ratio = 1.0f / h_ratio;

        const float orig_x_center = x_center * inv_w_ratio;
        const float orig_y_center = y_center * inv_h_ratio;
        const float orig_width = width * inv_w_ratio;
        const float orig_height = height * inv_h_ratio;

        // 边界计算优化
        const float half_width = orig_width * 0.5f;
        const float half_height = orig_height * 0.5f;

        const int left = std::max(0, static_cast<int>(orig_x_center - half_width));
        const int top = std::max(0, static_cast<int>(orig_y_center - half_height));
        const int right = std::min(original_img.cols, static_cast<int>(orig_x_center + half_width));
        const int bottom = std::min(original_img.rows, static_cast<int>(orig_y_center + half_height));

        const int box_width = right - left;
        const int box_height = bottom - top;

        // 有效性检查
        if (box_width <= 0 || box_height <= 0) continue;

        boxes.emplace_back(std::vector<float>{
            static_cast<float>(left), static_cast<float>(top),
                static_cast<float>(box_width), static_cast<float>(box_height)
        });
        scores.emplace_back(max_score);
        class_ids.emplace_back(max_idx);
    }

    // 第二阶段：NMS处理
    cv_boxes.reserve(boxes.size());
    for (const auto& box : boxes) {
        cv_boxes.emplace_back(
            static_cast<int>(box[0]), static_cast<int>(box[1]),
            static_cast<int>(box[2]), static_cast<int>(box[3])
        );
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(cv_boxes, scores, confidence_thres_, iou_thres_, indices);

    // 第三阶段：结果构建和DK处理
    results.reserve(indices.size());

    for (const int idx : indices) {
        DetectionResult res;
        const int class_id = class_ids[idx];

        // 类别名称设置
        if (class_id >= 0 && class_id < static_cast<int>(Inception_TRT_DLL::CLASS_NAMES.size())) {
            res.class_name = Inception_TRT_DLL::CLASS_NAMES[class_id];
        }
        else {
            res.class_name = "UnknownID_" + std::to_string(class_id);
        }

        res.bbox = cv_boxes[idx];
        res.confidence = scores[idx];
        res.area = res.bbox.width * res.bbox.height;

        // 高性能DK类处理
        if (res.class_name.find("DK") != std::string::npos) {
            process_dk_detection_optimized(res, original_img);
        }

        results.emplace_back(std::move(res));
    }

    return results;
}
INCEPTIONSERVICEDLL_API std::vector<DetectionResult> YOLO12TRTInfer::infer_with_optimized_dk(const std::string& image_path) {
    cv::Mat original_img;
    float h_ratio, w_ratio;

    // 使用现有的预处理
    std::vector<float> input_tensor = preprocess(image_path, original_img, h_ratio, w_ratio);

    // CUDA内存传输
    cudaError_t cuda_status = cudaMemcpyAsync(input_device_buffer_, input_tensor.data(),
        input_size_, cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy input data to GPU: " << cudaGetErrorString(cuda_status) << std::endl;
        return {};
    }

    // 设置输入形状
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 1;
    input_dims.d[1] = 3;
    input_dims.d[2] = input_height_;
    input_dims.d[3] = input_width_;
    context_->setInputShape(input_tensor_name_.c_str(), input_dims);

    // 设置张量地址
    context_->setTensorAddress(input_tensor_name_.c_str(), input_device_buffer_);
    context_->setTensorAddress(output_tensor_name_.c_str(), output_device_buffer_);

    // 执行推理
    bool status = context_->enqueueV3(stream_);
    if (!status) {
        std::cerr << "TensorRT inference failed" << std::endl;
        return {};
    }

    // 输出数据传输
    std::vector<float> output_data(output_size_ / sizeof(float));
    cuda_status = cudaMemcpyAsync(output_data.data(), output_device_buffer_,
        output_size_, cudaMemcpyDeviceToHost, stream_);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy output data from GPU: " << cudaGetErrorString(cuda_status) << std::endl;
        return {};
    }

    cudaStreamSynchronize(stream_);

    // 获取输出维度
    nvinfer1::Dims output_dims = engine_->getTensorShape(output_tensor_name_.c_str());
    int orig_rows = static_cast<int>(output_dims.d[1]);
    int orig_cols = static_cast<int>(output_dims.d[2]);

    // 优化的转置操作
    std::vector<float> transposed_output(orig_cols * orig_rows);

    // 使用OpenMP并行化（如果可用）
#pragma omp parallel for
    for (int i = 0; i < orig_rows; ++i) {
        for (int j = 0; j < orig_cols; ++j) {
            transposed_output[j * orig_rows + i] = output_data[i * orig_cols + j];
        }
    }

    // 使用优化的后处理
    return postprocess_optimized(transposed_output, orig_cols, orig_rows, h_ratio, w_ratio, original_img);
}
INCEPTIONSERVICEDLL_API std::string YOLO12TRTInfer::predict_with_optimized_dk(const std::string& image_path,
    bool visual, bool show_score, bool show_class, bool save_or_not) {

    std::vector<DetectionResult> results = infer_with_optimized_dk(image_path);

    if (results.empty()) {
        return R"([{"class_name": "ZC"}])";
    }

    // 只在需要时读取图像用于可视化
    if (visual || save_or_not) {
        cv::Mat img = InceptionUtils::imread_unicode(image_path, cv::IMREAD_COLOR);

        for (const auto& res : results) {
            draw_box(img, res, show_score, show_class);
        }

        if (save_or_not) {
            fs::path p(image_path);
            std::string save_dir = p.parent_path().string() + "//detection_result";
            fs::create_directories(save_dir);
            std::string save_file = save_dir + "//" + p.filename().string();
            InceptionUtils::imwrite_unicode(save_file, img);
        }

        if (visual) {
            std::string img_name = fs::path(image_path).filename().string();
            std::string window_name = "YOLO12_TensorRT_Detection_" + img_name;
            cv::namedWindow(window_name, cv::WINDOW_NORMAL);
            cv::imshow(window_name, img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }

    return detection_results_to_string(results);
}


INCEPTIONSERVICEDLL_API InceptionTRT::InceptionTRT(const std::string& classification_engine,
    const std::string& detection_engine,
    cv::Size classification_input_size,
    cv::Size detection_input_size,
    float confidence_thres,
    float iou_thres,
    int stretch_ratio)
    : classification_input_size_(classification_input_size),
    detection_input_size_(detection_input_size),
    confidence_thres_(confidence_thres),
    iou_thres_(iou_thres),
    stretch_ratio_(stretch_ratio) {

    try {
        // 初始化分类器
        classifier_ = std::make_unique<Classifier_TRT_Infer>(classification_engine, classification_input_size);
        //std::cout << "分类器初始化成功: " << classification_engine << std::endl;

        // 初始化检测器
        detector_ = std::make_unique<YOLO12TRTInfer>(detection_engine, detection_input_size, confidence_thres, iou_thres);
        //std::cout << "检测器初始化成功: " << detection_engine << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "InceptionTRT 初始化失败: " << e.what() << std::endl;
        throw;
    }
}

INCEPTIONSERVICEDLL_API InceptionTRT::~InceptionTRT() {
    // 智能指针会自动清理资源
}

INCEPTIONSERVICEDLL_API std::vector<InceptionResult> InceptionTRT::process(const std::string& image_path,
    const int CROP_WIDE,
    const int CROP_THRESHOLD,
    const std::string CENTER_LIMIT,
    const int LIMIT_AREA,
    const std::string& temp_output_path,
    bool output_stretched_images) {
    std::vector<InceptionResult> results;
    fs::path original_path(image_path);
    std::string base_ = original_path.stem().string();
    std::string ext_ = original_path.extension().string();
    std::string parent_dir_ = original_path.parent_path().string();
    try {
        //第一步：切分图像并进行图像拉伸
        cv::Mat rail_image = railHeadAreaCROP(image_path, CROP_WIDE, CROP_THRESHOLD, CENTER_LIMIT, LIMIT_AREA);
        std::vector<cv::Mat> stretched_images = processImageStretching(image_path, rail_image, temp_output_path, output_stretched_images);
        if (stretched_images.empty()) {
            throw std::runtime_error("图像拉伸处理失败，没有生成拉伸图像");
        }

        // 第二步：对拉伸后的两部分图像进行分类并根据分类结果进行检测返回检测结果
        
        int total_pieces_ = stretched_images.size();
        int stretched_id_ = 1;

        for (const auto& stretched_piece : stretched_images) {
            InceptionResult result;
            std::string stretched_piece_name = temp_output_path + "\\" + base_ + "_" +
                std::to_string(stretched_id_) + "of" + std::to_string(total_pieces_);
            std::string stretched_detection_save_path = temp_output_path + "\\Detection_Anomaly\\" + base_ + "_" +
                std::to_string(stretched_id_) + "of" + std::to_string(total_pieces_) + ext_;
            result.img_path = stretched_detection_save_path;  // 完整路径
            result.img_name = stretched_piece_name;  // 文件名

            ClassificationResult classification_result = classifyImage(stretched_piece, stretched_piece_name);
            //std::cout << "分类结果: class_id=" << classification_result.class_id
            //    << ", class_name=" << classification_result.class_name
            //    << ", confidence=" << classification_result.confidence << std::endl;
            if (classification_result.class_id == 1) {
                //std::cout << "分类结果为1，开始检测处理..." << std::endl;
                std::vector<DetectionResult> detection_results;
                // 执行检测
                if (TestModel_Flag) {
                     detection_results = detectImage_with_TestModel_Flag(stretched_piece, stretched_detection_save_path);
                }
                else {
                     detection_results = detectImage(stretched_piece, stretched_detection_save_path);
                }


                // 返回检测结果
                result.result_type = InceptionResult::DETECTION;
                result.detectionresults = detection_results;
                result.classificationresult = classification_result;

                //std::cout << "检测完成，发现 " << detection_results.size() << " 个目标" << std::endl;
            }
            else {
                //std::cout << "分类结果不为1，返回分类结果" << std::endl;

                // 返回分类结果
                result.result_type = InceptionResult::CLASSIFICATION;
                result.classificationresult = classification_result;
            }
            //result添加到results中
            results.push_back(std::move(result));  // 使用 move 优化
            stretched_id_++;
        }

    }
    catch (const std::exception& e) {
        std::cerr << "InceptionTRT::process 处理失败: " << e.what() << std::endl;
        // 返回错误结果
        InceptionResult result;
        result.result_type = InceptionResult::CLASSIFICATION;
        result.img_name = base_;
        result.img_path = image_path;
        result.classificationresult.class_id = -1;
        result.classificationresult.class_name = "ERROR";
        result.classificationresult.confidence = 0.0f;
        results.push_back(result);
    }

    return results;
}
INCEPTIONSERVICEDLL_API cv::Mat InceptionTRT::railHeadAreaCROP(const std::string& image_path, const int CROP_WIDE, const int CROP_THRESHOLD, const std::string CENTER_LIMIT, const int LIMIT_AREA) {
    try {
        cv::Mat railHeadArea;
        cv::Mat original_img = InceptionUtils::imread_unicode(image_path, cv::IMREAD_COLOR);
        if (original_img.empty()) {
            throw std::runtime_error("无法读取图像: " + image_path);
        }
        std::string filename = fs::path(image_path).filename().string();
        int crop_threshold = CROP_THRESHOLD;
        int crop_kernel_size = 5;
        int crop_wide = CROP_WIDE;
        bool center_limit = (CENTER_LIMIT == "true");
        int limit_area = LIMIT_AREA;

        railHeadArea = Inception_TRT_DLL::CropRailhead(image_path, crop_threshold, crop_kernel_size,crop_wide,center_limit,limit_area);

        return railHeadArea;
    }
    catch (const std::exception& e) {
        std::cerr << "图像拉伸处理失败：" << e.what() << std::endl;
    }



}

INCEPTIONSERVICEDLL_API std::vector<cv::Mat> InceptionTRT::processImageStretching(const std::string& image_path,
    cv::Mat railHeadArea,
    const std::string& output_path,
    bool output_files) {
    std::vector<cv::Mat> stretched_images;
    try {
        std::string filename = fs::path(image_path).filename().string();
        stretched_images = Inception_TRT_DLL::StretchAndSplit(
            railHeadArea,
            filename,
            output_files,
            output_path,
            stretch_ratio_
        );
    }
    catch (const std::exception& e) {
        std::cerr << "图像拉伸处理失败：" << e.what() << std::endl;
    }


    return stretched_images;
}

INCEPTIONSERVICEDLL_API ClassificationResult InceptionTRT::classifyImage(const cv::Mat& image, const std::string& image_name) {
    ClassificationResult C_Result_;
    C_Result_.class_id = -1;
    C_Result_.class_name = "UNKNOWN";
    C_Result_.confidence = 0.0f;

    try {
        if (image.empty()) {

            throw std::runtime_error("输入图像为空");
        }

        C_Result_ = classifier_->predict(image);
    }
    catch (const std::exception& e) {
        std::cerr << "分类处理失败: " << e.what() << std::endl;
        C_Result_.confidence = 0.0f;
        C_Result_.class_id = -1;
        C_Result_.class_name = "ERROR";
    }
    return C_Result_;
}

INCEPTIONSERVICEDLL_API std::vector<DetectionResult> InceptionTRT::detectImage(const cv::Mat& image, const std::string detection_save_path) {
    std::vector<DetectionResult> detectionResults_;
    try {
        cv::Mat mutable_image = image.clone();
        detectionResults_ = detector_->predict(
            mutable_image,
            false,                  // visual - 禁用可视化显示
            true,                   // show_score
            true,                   // show_class  
            true,                  // save_or_not - 禁用文件保存
            detection_save_path     // img_path
        );
        //std::vector<DetectionResult> predict(cv::Mat& img, bool visual = false, bool show_score = true, bool show_class = true, bool save_or_not = false, std::string img_path);
        //std::cout << "检测片段 " << ": 发现 " << detectionResults_.size() << " 个目标" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Mat InceptionTRT::detectImage检测程式失败" << std::endl;
    }
    return detectionResults_;

}

INCEPTIONSERVICEDLL_API std::vector<DetectionResult> InceptionTRT::detectImage_with_TestModel_Flag(const cv::Mat& image, const std::string& detection_save_path) {
    std::vector<DetectionResult> detectionResults_;

    try {
        // 输入验证
        if (image.empty()) {
            std::cerr << "[ERROR] InceptionTRT::detectImage - 输入图像为空" << std::endl;
            std::cerr << "  - 保存路径: " << detection_save_path << std::endl;
            return detectionResults_;
        }

        if (!detector_) {
            std::cerr << "[ERROR] InceptionTRT::detectImage - 检测器未初始化" << std::endl;
            std::cerr << "  - 图像尺寸: " << image.cols << "x" << image.rows << std::endl;
            std::cerr << "  - 保存路径: " << detection_save_path << std::endl;
            return detectionResults_;
        }

        // 打印调试信息
        if (TestModel_Flag) {
            std::cout << "[DEBUG] InceptionTRT::detectImage - 开始检测" << std::endl;
            std::cout << "  - 图像尺寸: " << image.cols << "x" << image.rows << std::endl;
            std::cout << "  - 图像通道: " << image.channels() << std::endl;
            std::cout << "  - 图像类型: " << image.type() << std::endl;
            std::cout << "  - 保存路径: " << detection_save_path << std::endl;
        }

        // 创建可修改的图像副本
        cv::Mat mutable_image = image.clone();

        // 执行检测
        detectionResults_ = detector_->predict(
            mutable_image,
            true,                   // visual
            true,                   // show_score
            true,                   // show_class
            true,                   // save_or_not
            detection_save_path     // img_path
        );

        // 输出结果信息
        if (TestModel_Flag) {
            std::cout << "[INFO] InceptionTRT::detectImage - 检测完成" << std::endl;
            std::cout << "  - 检测到目标数量: " << detectionResults_.size() << std::endl;
            std::cout << "  - 保存路径: " << detection_save_path << std::endl;
        }
        // 详细输出检测结果
        if (TestModel_Flag && !detectionResults_.empty()) {
            std::cout << "[DEBUG] 检测结果详情:" << std::endl;
            for (size_t i = 0; i < detectionResults_.size(); ++i) {
                const auto& result = detectionResults_[i];
                std::cout << "  [" << i << "] 类别: " << result.class_name
                    << ", 置信度: " << std::fixed << std::setprecision(3) << result.confidence
                    << ", 区域: [" << result.bbox.x << ", " << result.bbox.y
                    << ", " << result.bbox.width << ", " << result.bbox.height << "]"
                    << ", 面积: " << result.area << std::endl;
            }
        }

    }
    catch (const cv::Exception& cv_e) {
        std::cerr << "[ERROR] InceptionTRT::detectImage - OpenCV异常" << std::endl;
        std::cerr << "  - 错误代码: " << cv_e.code << std::endl;
        std::cerr << "  - 错误信息: " << cv_e.what() << std::endl;
        std::cerr << "  - 文件: " << cv_e.file << ":" << cv_e.line << std::endl;
        std::cerr << "  - 函数: " << cv_e.func << std::endl;
        std::cerr << "  - 图像信息: " << image.cols << "x" << image.rows
            << ", 通道: " << image.channels() << ", 类型: " << image.type() << std::endl;
        std::cerr << "  - 保存路径: " << detection_save_path << std::endl;
    }
    catch (const std::runtime_error& rt_e) {
        std::cerr << "[ERROR] InceptionTRT::detectImage - 运行时异常" << std::endl;
        std::cerr << "  - 错误信息: " << rt_e.what() << std::endl;
        std::cerr << "  - 可能原因: TensorRT推理失败、CUDA内存不足、模型加载失败" << std::endl;
        std::cerr << "  - 图像信息: " << image.cols << "x" << image.rows
            << ", 通道: " << image.channels() << ", 类型: " << image.type() << std::endl;
        std::cerr << "  - 保存路径: " << detection_save_path << std::endl;
    }
    catch (const std::bad_alloc& ba_e) {
        std::cerr << "[ERROR] InceptionTRT::detectImage - 内存分配失败" << std::endl;
        std::cerr << "  - 错误信息: " << ba_e.what() << std::endl;
        std::cerr << "  - 可能原因: 系统内存不足、GPU显存不足" << std::endl;
        std::cerr << "  - 图像信息: " << image.cols << "x" << image.rows
            << ", 通道: " << image.channels() << ", 类型: " << image.type() << std::endl;
        std::cerr << "  - 保存路径: " << detection_save_path << std::endl;
    }
    catch (const std::invalid_argument& ia_e) {
        std::cerr << "[ERROR] InceptionTRT::detectImage - 无效参数异常" << std::endl;
        std::cerr << "  - 错误信息: " << ia_e.what() << std::endl;
        std::cerr << "  - 可能原因: 参数格式错误、路径无效" << std::endl;
        std::cerr << "  - 图像信息: " << image.cols << "x" << image.rows
            << ", 通道: " << image.channels() << ", 类型: " << image.type() << std::endl;
        std::cerr << "  - 保存路径: " << detection_save_path << std::endl;
    }
    catch (const std::filesystem::filesystem_error& fs_e) {
        std::cerr << "[ERROR] InceptionTRT::detectImage - 文件系统异常" << std::endl;
        std::cerr << "  - 错误信息: " << fs_e.what() << std::endl;
        std::cerr << "  - 错误代码: " << fs_e.code().value() << std::endl;
        std::cerr << "  - 路径1: " << fs_e.path1() << std::endl;
        std::cerr << "  - 路径2: " << fs_e.path2() << std::endl;
        std::cerr << "  - 保存路径: " << detection_save_path << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] InceptionTRT::detectImage - 标准异常" << std::endl;
        std::cerr << "  - 异常类型: " << typeid(e).name() << std::endl;
        std::cerr << "  - 错误信息: " << e.what() << std::endl;
        std::cerr << "  - 图像信息: " << image.cols << "x" << image.rows
            << ", 通道: " << image.channels() << ", 类型: " << image.type() << std::endl;
        std::cerr << "  - 保存路径: " << detection_save_path << std::endl;

        // 尝试获取更多系统信息
        try {
            std::cerr << "  - 当前工作目录: " << std::filesystem::current_path() << std::endl;
        }
        catch (...) {
            std::cerr << "  - 无法获取当前工作目录" << std::endl;
        }
    }
    catch (...) {
        std::cerr << "[ERROR] InceptionTRT::detectImage - 未知异常" << std::endl;
        std::cerr << "  - 异常类型: 未知类型" << std::endl;
        std::cerr << "  - 图像信息: " << image.cols << "x" << image.rows
            << ", 通道: " << image.channels() << ", 类型: " << image.type() << std::endl;
        std::cerr << "  - 保存路径: " << detection_save_path << std::endl;
        std::cerr << "  - 建议: 检查CUDA驱动、TensorRT版本、模型文件" << std::endl;
    }

    return detectionResults_;
}


INCEPTIONSERVICEDLL_API std::string InceptionTRT::getResultAsJson(const InceptionResult& result) {
    nlohmann::json json_result;

    try {
        // 添加基本图像信息
        json_result["img_name"] = result.img_name;
        json_result["img_path"] = result.img_path;

        if (result.result_type == InceptionResult::CLASSIFICATION) {
            json_result["type"] = "classification";
            json_result["classification"] = {
                {"class_id", result.classificationresult.class_id},
                {"class_name", result.classificationresult.class_name},
                {"confidence", result.classificationresult.confidence}
            };
        }
        else if (result.result_type == InceptionResult::DETECTION) {
            json_result["type"] = "detection";

            // 包含分类信息（检测前的分类结果）
            json_result["classification"] = {
                {"class_id", result.classificationresult.class_id},
                {"class_name", result.classificationresult.class_name},
                {"confidence", result.classificationresult.confidence}
            };

            // 检测结果数组
            nlohmann::json detections_array = nlohmann::json::array();
            for (const auto& detection : result.detectionresults) {
                nlohmann::json det_obj;
                det_obj["class_name"] = detection.class_name;
                det_obj["bbox"] = {
                    {"x", detection.bbox.x},
                    {"y", detection.bbox.y},
                    {"width", detection.bbox.width},
                    {"height", detection.bbox.height}
                };
                det_obj["confidence"] = detection.confidence;
                det_obj["area"] = detection.area;
                det_obj["area_contour"] = detection.area_contour;

                // 添加轮廓信息（如果存在）
                if (!detection.contours.empty()) {
                    nlohmann::json contours_json = nlohmann::json::array();
                    for (const auto& contour : detection.contours) {
                        nlohmann::json contour_json = nlohmann::json::array();
                        for (const auto& pt : contour) {
                            contour_json.push_back({
                                {"x", pt.x},
                                {"y", pt.y}
                                });
                        }
                        contours_json.push_back(contour_json);
                    }
                    det_obj["contours"] = contours_json;
                }

                detections_array.push_back(det_obj);
            }
            json_result["detections"] = detections_array;
            json_result["detection_count"] = result.detectionresults.size();
        }
        else {
            // 处理未知类型
            json_result["type"] = "unknown";
            json_result["error"] = "Unknown result type";
        }

        // 添加时间戳 - 修复安全警告
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        // 使用线程安全的localtime_s
        std::tm tm_buf;
#ifdef _WIN32
        localtime_s(&tm_buf, &time_t);
#else
        localtime_r(&time_t, &tm_buf);
#endif

        std::stringstream ss;
        ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
        json_result["timestamp"] = ss.str();

    }
    catch (const std::exception& e) {
        std::cerr << "JSON序列化失败: " << e.what() << std::endl;

        // 返回错误信息的JSON
        nlohmann::json error_result;
        error_result["type"] = "error";
        error_result["error"] = e.what();
        error_result["img_name"] = result.img_name;
        error_result["img_path"] = result.img_path;

        return error_result.dump();
    }

    return json_result.dump(4);  // 使用4个空格的缩进格式化输出
}