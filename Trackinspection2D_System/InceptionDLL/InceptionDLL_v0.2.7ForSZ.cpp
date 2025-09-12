// InceptionDLL.cpp : 定义 DLL 的导出函数。
//

#include "pch.h"
#include "framework.h"
#include "InceptionDLL_v0.2.7ForSZ.h"
#include "InceptionUtils_v0.2.7ForSZ.h"

#include <algorithm>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

bool TestModel_Flag = false;
bool Data_Collector = false;

namespace fs = std::filesystem;
using namespace InceptionUtils;


INCEPTIONDLL_API cv::Mat selectThresholdAdaptive(const cv::Mat& gray_inv) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray_inv, mean, stddev);

    double img_mean = mean[0];
    double img_std = stddev[0];

    cv::Mat result;

    // 根据图像特性选择最佳阈值方法
    if (img_std > 200) { // 高对比度，OTSU效果好
        cv::threshold(gray_inv, result, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    }
    else if (img_std < 20) { // 低对比度，使用Triangle
        cv::threshold(gray_inv, result, 0, 255, cv::THRESH_BINARY + cv::THRESH_TRIANGLE);
    }
    else { // 中等对比度，尝试组合
        cv::Mat otsu, triangle;
        cv::threshold(gray_inv, otsu, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        cv::threshold(gray_inv, triangle, 0, 255, cv::THRESH_BINARY + cv::THRESH_TRIANGLE);
        cv::bitwise_and(otsu, triangle, result);
    }

    return result;
}
INCEPTIONDLL_API std::string detection_results_to_string(const std::vector<DetectionResult>& results) {
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
            obj["area_contour"] = res.area_contour;
        }

        arr.push_back(obj);
    }
    return arr.dump();
}
namespace InceptionDLL {
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
    std::unordered_map<std::string, cv::Mat> chinese_labels;
    // 预先创建中文标签图片
    void createChineseLabels() {
        cv::Mat dk_label = cv::Mat::zeros(20, 40, CV_8UC3);
        cv::putText(dk_label, "掉块", cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        chinese_labels["DK"] = dk_label;

        cv::Mat cs_label = cv::Mat::zeros(20, 40, CV_8UC3);
        cv::putText(cs_label, "擦伤", cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        chinese_labels["CS"] = cs_label;
    }
    const std::vector<std::string> CLASS_NAMES = {
        // "DK_A", "DK_B", "DK_C", "CS_A", "CS_B", "CS_C"
        "DK_A", "DK_B", "DK_C", "CS_A", "CS_B", "CS_C","WZ","GF","HF"
    };
    // Opencv中以BGR为颜色通道顺序
    const std::map<std::string, cv::Scalar> CLASS_COLORS = {
        {"DK_A", cv::Scalar(0, 0, 204)},
        {"DK_B", cv::Scalar(0, 255, 255)},
        {"DK_C", cv::Scalar(102, 204, 0)},
        {"CS_A", cv::Scalar(0, 0, 204)},
        {"CS_B", cv::Scalar(0, 255, 255)},
        {"CS_C", cv::Scalar(0, 204, 0)},
        {"WZ",   cv::Scalar(230,230,255)},
        {"GF",   cv::Scalar(255,26,0)},
        {"HF",   cv::Scalar(125,125,125)}
    };
}
namespace InceptionDLL {


    INCEPTIONDLL_API cv::Mat CropRailhead(
        const std::string& img_path, int crop_threshold, int crop_kernel_size, int crop_wide, bool center_limit, int limit_area)
    {
        cv::Mat img = imread_unicode(img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "CropRailhead: 图像读取失败: " << img_path << std::endl;
            return cv::Mat();
        }
        return RailheadCropHighlightCenterArea(img, crop_threshold, crop_kernel_size, crop_wide, center_limit,limit_area);
    }
    INCEPTIONDLL_API cv::Mat RailheadCropHighlightCenterArea(
        const cv::Mat& img, int threshold, int kernel_size, int crop_wide, bool center_limit = true, int limit_area = 50)
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
        if (!contours.empty()) {
            auto largest = std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) < cv::contourArea(b);
                });
            cv::Rect bbox = cv::boundingRect(*largest);
            crod_m = bbox.x + bbox.width / 2;
            if (center_limit) {
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
    INCEPTIONDLL_API std::vector<std::string> StretchAndSplit(
        const cv::Mat& cropped,
        const std::string& cropped_name,
        const bool& output_or_not,
        const std::string& stretch_output_path,
        int stretch_ratio)

    {
        std::vector<std::string> stretch_piece_paths;
        // 检查输入图像是否有效
        if (cropped.empty() || cropped.rows <= 0 || cropped.cols <= 0) {
            std::cerr << "StretchAndSplit: 输入图像无效" << std::endl;
            return stretch_piece_paths;
        }
        int orig_h = cropped.rows, orig_w = cropped.cols;
        int new_h = orig_h * stretch_ratio;
        cv::Mat stretched;
        cv::resize(cropped, stretched, cv::Size(orig_w, new_h), 0, 0, cv::INTER_LINEAR);

        int count = new_h / orig_h;
        int rem = new_h % orig_h;
        std::string base = fs::path(cropped_name).stem().string();
        std::string ext = fs::path(cropped_name).extension().string();
        
        if (output_or_not) fs::create_directories(stretch_output_path);
        for (int i = 0; i < count; ++i) {
            cv::Mat piece = stretched.rowRange(i * orig_h, (i + 1) * orig_h);
            std::string out_name = base + "_" + std::to_string(count + (rem ? 1 : 0)) + "of" + std::to_string(i + 1) + ext;
            std::string out_path = stretch_output_path + "/" + out_name;
            if (output_or_not) imwrite_unicode(out_path, piece);
            stretch_piece_paths.push_back(out_path);
        }
        if (rem) {
            cv::Mat piece = stretched.rowRange(count * orig_h, new_h);
            std::string out_name = base + "_" + std::to_string(count + 1) + "of" + std::to_string(count + 1) + ext;
            std::string out_path = stretch_output_path + "/" + out_name;
            if (output_or_not) imwrite_unicode(out_path, piece);
            stretch_piece_paths.push_back(out_path);
        }
        return stretch_piece_paths;
    }

    INCEPTIONDLL_API std::string ClassPredictOnnx(
        Ort::Session& session, const cv::Mat& img_input, int img_size)
    {
        cv::Mat img;
        if (img_input.empty()) {
            std::cout << "ClassPredictOnnx 获取的图像为空" << std::endl;
            return "110 Unknown";
        }
        cv::resize(img_input, img, cv::Size(img_size, img_size));
        img.convertTo(img, CV_32F, 1.0 / 255);
        std::vector<float> input_tensor_values;
        input_tensor_values.reserve(img_size * img_size * 3);
        cv::Mat channels[3];
        cv::split(img, channels);
        for (int c = 2; c >= 0; c--) {
            const float* channel_data = channels[c].ptr<float>();
            input_tensor_values.insert(input_tensor_values.end(), channel_data, channel_data + img_size * img_size);
        }
        std::vector<int64_t> input_shape = { 1, 3, img_size, img_size };
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());
        const char* input_names[] = { "input" };
        const char* output_names[] = { "output" };
        try {
            auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
            const float* output_data = output_tensors[0].GetTensorData<float>();
            size_t output_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
            int max_index = 5;
            float max_value = output_data[5];
            for (size_t i = 0; i < output_count; i++) {

                // 剔除 ~i=2BM~ i=3 HF和i=6GF ~i=7GD~ 的影响
                if (i == 3 || i == 6) {
                    continue;
                }
                if (output_data[i] > max_value) {
                    max_value = output_data[i];
                    max_index = static_cast<int>(i);
                    /*        
                    {0, "YC"},
                    {1, "DK"},
                    {2, "BM"},
                    {3, "HF"},
                    {4, "CS"},
                    {5, "ZC"},
                    {6, "GF"},
                    {7, "GD"}
                    */
                }
            }
            // 添加置信度过滤逻辑
            float confidence_threshold = 0.3; // 根据实际情况调整
            if (max_index == 2) {
                confidence_threshold = 0.2; // BM类别的阈值更低
            }
            if (max_value < confidence_threshold) {
                return "ZC";
            }
            auto it = InceptionDLL::classes_lable_map.find(max_index);
            return (it != InceptionDLL::classes_lable_map.end()) ? it->second : "111 Unknown";
        }
        catch (...) {
            return "112 Unknown";
        }
    }

    INCEPTIONDLL_API std::string ClassifyImage(
        Ort::Session& classify_session,
        const std::string& img_path,
        int img_size,
        const std::string& temp_path)
    {
        cv::Mat img = imread_unicode(img_path);
        if (img.empty()) {
            std::cout << "ClassifyImage 获取的图像为空: " << img_path << std::endl;
            return "110 Unknown";
        }
        return ClassPredictOnnx(classify_session, img, img_size);
    }

    INCEPTIONDLL_API std::string DetectionOnnx(
        YOLO12Infer& detector, const cv::Mat& img_input)
    {
        std::string temp_path = "temp_detection.jpg";
        imwrite_unicode(temp_path, img_input);
        return detector.predict(temp_path, false, true, true, true);
    }

    INCEPTIONDLL_API std::string DetectImage(
        YOLO12Infer& detector,
        const std::string& img_path,
        const std::string& temp_path)
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

            //======正常的逻辑======
            //if (!det_results.empty() && det_results.size() == 1) {
            //    return "['class_name': 'ZC']";
            //}
            //else if (det_results.empty()) {
            //    return "['class_name': 'ZC']";
            //}
            //else return det_results;

            //======检查是否至少包含"DK_A", "DK_B", "DK_C", "CS_A", "CS_B", "CS_C" 或 "WZ" 这几类，如果不包含则返回 "['class_name': 'ZC']"========
            if (det_results.empty()) {
                return R"([{"class_name":"ZC"}])";
            }

            // 解析 JSON 数组形式的检测结果
            try {
                auto json_results = nlohmann::json::parse(det_results);
                bool containsDesired = false;
                for (const auto& item : json_results) {
                    std::string cls = item.value("class_name", "");
                    if (cls == "DK_A" || cls == "DK_B" || cls == "DK_C" ||
                        cls == "CS_A" || cls == "CS_B" || cls == "CS_C" || cls == "WZ")
                    {
                        containsDesired = true;
                        break;
                    }
                }
                if (!containsDesired) {
                    return R"([{"class_name":"ZC"}])";
                }
            }
            catch (...) {
                // 解析失败也返回默认值
                return R"([{"class_name":"ZC"}])";
            }

            return det_results;
            //======检查是否至少包含"DK_A", "DK_B", "DK_C", "CS_A", "CS_B", "CS_C" 或 "WZ" 这几类，如果不包含则返回 "['class_name': 'ZC']"========


        }
        catch (const std::exception& e) {
            std::cerr << "DetectImage 异常: " << e.what() << std::endl;
            return R"([{"class_name":"ZC"}])";
        }
        catch (...) {
            std::cerr << "DetectImage 未知异常" << std::endl;
            return R"([{"class_name":"ZC"}])";
        }
    }

} // namespace InceptionDLL
void print_detection_box_info(const float* data, int cols,int max_class_idx, float max_class_score, float confidence) {
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "框原始数据: [";
    for (int i = 0; i < std::min(10, cols); ++i) {
        std::cout << data[i];
        if (i < std::min(10, cols) - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    float x = data[0], y = data[1], w = data[2], h = data[3];
    std::cout << "坐标 (原始): x=" << x << ", y=" << y << ", w=" << w << ", h=" << h << std::endl;

    float stride = 16.0f;
    float x_scaled = x * stride, y_scaled = y * stride, w_scaled = w * stride, h_scaled = h * stride;
    std::cout << "坐标 (stride=" << stride << "): x=" << x_scaled << ", y=" << y_scaled
        << ", w=" << w_scaled << ", h=" << h_scaled << std::endl;

    //std::cout << "置信度: obj=" << obj_conf << ", class=" << max_class_score << ", final=" << confidence << std::endl;
    if (max_class_idx >= 0 && max_class_idx < static_cast<int>(InceptionDLL::CLASS_NAMES.size())) {
        std::cout << "类别: id=" << max_class_idx << ", name=" << InceptionDLL::CLASS_NAMES[max_class_idx] << std::endl;
    }
    else {
        std::cout << "类别: id=" << max_class_idx << ", INVALID" << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;
}

INCEPTIONDLL_API YOLO12Infer::YOLO12Infer(const std::string& onnx_model,
    cv::Size input_image_size,
    float confidence_thres,
    float iou_thres,
    bool use_gpu)
    : input_image_size_(input_image_size),
    confidence_thres_(confidence_thres),
    iou_thres_(iou_thres),
    env_(ORT_LOGGING_LEVEL_WARNING, "YOLO12"),
    session_options_(),
    session_(nullptr)
{
    session_options_.SetIntraOpNumThreads(1);
    if (use_gpu) {
#ifdef _WIN32
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0);
#endif
    }
#ifdef _WIN32
    std::wstring w_model = std::filesystem::path(onnx_model).wstring();
    session_ = Ort::Session(env_, w_model.c_str(), session_options_);
#else
    session_ = Ort::Session(env_, onnx_model.c_str(), session_options_);
#endif
    // 获取输入尺寸
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session_.GetInputNameAllocated(0, allocator);
    auto input_type_info = session_.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_shape = input_tensor_info.GetShape();
    input_width_ = static_cast<int>(input_shape[2]);
    input_height_ = static_cast<int>(input_shape[3]);
}

INCEPTIONDLL_API cv::Mat YOLO12Infer::letterbox(const cv::Mat& img, float& h_ratio, float& w_ratio) {
    cv::Size target_size = input_image_size_;

    // 检查输入图像尺寸，防止除零错误
    if (img.rows <= 0 || img.cols <= 0) {
        std::cerr << "错误：输入图像尺寸无效 - 宽度: " << img.cols << ", 高度: " << img.rows << std::endl;
        // 返回一个空白图像并设置安全的缩放比例
        cv::Mat empty(target_size, CV_8UC3, cv::Scalar(0, 0, 0));
        h_ratio = w_ratio = 1.0f;
        return empty;
    }

    cv::Mat resized;
    cv::resize(img, resized, target_size, 0, 0, cv::INTER_LINEAR);
    h_ratio = static_cast<float>(target_size.height) / img.rows;
    w_ratio = static_cast<float>(target_size.width) / img.cols;
    return resized;
}
//INCEPTIONDLL_API cv::Mat YOLO12Infer::letterbox(const cv::Mat& img, float& h_ratio, float& w_ratio) {
//    cv::Size target_size = input_image_size_;
//
//    // 计算缩放比例，保持长宽比
//    float scale = std::min(
//        static_cast<float>(target_size.width) / img.cols,
//        static_cast<float>(target_size.height) / img.rows
//    );
//
//    // 计算新尺寸
//    int new_width = static_cast<int>(img.cols * scale);
//    int new_height = static_cast<int>(img.rows * scale);
//
//    // 计算填充
//    int pad_x = (target_size.width - new_width) / 2;
//    int pad_y = (target_size.height - new_height) / 2;
//
//    // 记录缩放和填充信息，用于后处理
//    h_ratio = scale;
//    w_ratio = scale;
//
//    // 调整图像大小
//    cv::Mat resized;
//    cv::resize(img, resized, cv::Size(new_width, new_height));
//
//    // 创建填充后的图像（灰色背景）
//    cv::Mat padded(target_size, resized.type(), cv::Scalar(114, 114, 114));
//
//    // 将调整大小的图像复制到填充图像的中心
//    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_width, new_height)));
//
//    return padded;
//}

INCEPTIONDLL_API std::vector<float> YOLO12Infer::preprocess(const std::string& image_path, cv::Mat& original_img, float& h_ratio, float& w_ratio) {
    original_img = InceptionUtils::imread_unicode(image_path, cv::IMREAD_COLOR);
    if (original_img.empty()) throw std::runtime_error("无法读取图像: " + image_path);

    if (TestModel_Flag ==true){
        // 打印原始图像信息
        std::cout << "C++ 原始图像信息: " << image_path << std::endl;
        std::cout << "原始图像尺寸: " << original_img.cols << "x" << original_img.rows << std::endl;
}
    // 从BGR转为RGB
    cv::Mat img_rgb;
    cv::cvtColor(original_img, img_rgb, cv::COLOR_BGR2RGB);

    // 调整大小
    cv::Mat img_resized = letterbox(img_rgb, h_ratio, w_ratio);

    if (TestModel_Flag == true) {
        // 打印letterbox后的图像尺寸
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
    //for (int c = 0; c < 3; ++c) {
    //    input_tensor.insert(input_tensor.end(), (float*)chw[c].datastart, (float*)chw[c].dataend);
    //}
    input_tensor.reserve(img_resized.rows * img_resized.cols * 3);

    // 按照R,G,B顺序添加通道数据，与Python保持一致
    for (int c = 0; c < 3; ++c) {
        const float* channel_data = chw[c].ptr<float>();
        input_tensor.insert(input_tensor.end(), channel_data, channel_data + img_resized.rows * img_resized.cols);
    }

    if (TestModel_Flag == true) {
        // 打印输入tensor的统计信息
        size_t tensor_size = input_tensor.size();
        float sum = 0.0f, min_val = FLT_MAX, max_val = -FLT_MAX;
        for (size_t i = 0; i < std::min(tensor_size, static_cast<size_t>(100)); ++i) {
            sum += input_tensor[i];
            min_val = std::min(min_val, input_tensor[i]);
            max_val = std::max(max_val, input_tensor[i]);
        }

        std::cout << "C++ 输入tensor统计 (前100个元素):" << std::endl;
        std::cout << "- 元素总数: " << tensor_size << std::endl;
        std::cout << "- 形状: [1, 3, " << img_resized.rows << ", " << img_resized.cols << "]" << std::endl;
        std::cout << "- 均值: " << (sum / std::min(tensor_size, static_cast<size_t>(100))) << std::endl;
        std::cout << "- 最小值: " << min_val << std::endl;
        std::cout << "- 最大值: " << max_val << std::endl;
        std::cout << "- 前10个值: ";
        for (size_t i = 0; i < std::min(tensor_size, static_cast<size_t>(10)); ++i) {
            std::cout << input_tensor[i] << " ";
        }
        std::cout << std::endl;
    }
    return input_tensor;
}

INCEPTIONDLL_API std::vector<DetectionResult> YOLO12Infer::postprocess(const std::vector<float>& output, int rows, int cols,
    float h_ratio, float w_ratio, const cv::Mat& original_img,const std::string& image_path) {
    std::vector<std::vector<float>> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<DetectionResult> results;

    //if (TestModel_Flag) {
    //    std::cout << "CLASS_NAMES数量: " << InceptionDLL::CLASS_NAMES.size() << std::endl;
    //    std::cout << "模型输出的形状：rows=" << rows << ", cols=" << cols << std::endl;
    //    std::cout << "输入图像尺寸：" << input_image_size_.width << "x" << input_image_size_.height << std::endl;
    //    std::cout << "原始图像尺寸: " << original_img.cols << "x" << original_img.rows << std::endl;
    //    std::cout << "缩放比例 w_ratio: " << w_ratio << ", h_ratio: " << h_ratio << std::endl;
    //}

    // 遍历每一行，每一行代表一个检测框
    for (int i = 0; i < rows; ++i) {
        const float* row = &output[i * cols];
        float max_class_score = 0.0f;
        int max_class_idx = -1;

        // 从索引4开始遍历，查找最大类别得分
        for (int j = 4; j < std::min(4 + static_cast<int>(InceptionDLL::CLASS_NAMES.size()), cols); ++j) {
            if (row[j] > max_class_score) {
                max_class_score = row[j];
                max_class_idx = j - 4;  // 转换为0-based类别索引
            }
        }

        // 直接使用最大类别得分作为置信度
        float confidence = max_class_score;

        if (TestModel_Flag && confidence >= confidence_thres_) {
            print_detection_box_info(row, cols, max_class_idx, max_class_score, confidence);
        }

        if (confidence >= confidence_thres_ && max_class_idx >= 0) {

            float x_center = row[0];
            float y_center = row[1];
            float width = row[2];
            float height = row[3];

            if (TestModel_Flag) {
                std::cout << "检测框 " << i << " - 边界框(应用stride前): ["
                    << row[0] << ", " << row[1] << ", " << row[2] << ", " << row[3] << "]" << std::endl;
                std::cout << "检测框 " << i << " - 边界框(应用stride后): ["
                    << x_center << ", " << y_center << ", " << width << ", " << height << "]" << std::endl;
            }

            // 将坐标从网络输入尺寸映射回原始图像尺寸
            float orig_x_center = x_center / w_ratio;
            float orig_y_center = y_center / h_ratio;
            float orig_width = width / w_ratio;
            float orig_height = height / h_ratio;
            if (TestModel_Flag && confidence >= confidence_thres_) {
    print_detection_box_info(row, cols, max_class_idx, max_class_score, confidence);
}
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
        cv_boxes.emplace_back((int)b[0], (int)b[1], (int)b[2], (int)b[3]);
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(cv_boxes, scores, confidence_thres_, iou_thres_, indices);
    
    int id = 0;
    // 创建结果对象
    for (int idx : indices) {
        DetectionResult res;
        int class_id = class_ids[idx];

        // 检查class_id是否在有效范围内
        if (class_id >= 0 && class_id < static_cast<int>(InceptionDLL::CLASS_NAMES.size())) {
            res.class_name = InceptionDLL::CLASS_NAMES[class_id];
        }
        else {
            std::cout << "警告：类别ID(" << class_id << ")超出范围" << std::endl;
            res.class_name = "UnknownID_" + std::to_string(class_id);
        }
        res.bbox = cv_boxes[idx];
        res.confidence = scores[idx];
        res.area = res.bbox.width * res.bbox.height;
        res.area_contour = 0.0f;  // 为所有检测结果设置默认值
        if (TestModel_Flag) {
            std::cout << "class_name:" << res.class_name << std::endl;
            std::cout << "bbox:" << res.bbox << std::endl;
        }
        // DK类处理

        if (res.class_name.find("DK") != std::string::npos) {
            try {
                if (TestModel_Flag) {
                    std::cout << "执行掉块类处理....." << std::endl;
                }
                // 裁剪检测区域
                int x1 = std::max(res.bbox.x, 0);
                int y1 = std::max(res.bbox.y, 0);
                int x2 = std::min(res.bbox.x + res.bbox.width, original_img.cols);
                int y2 = std::min(res.bbox.y + res.bbox.height, original_img.rows);
                cv::Mat crop = original_img(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();

                // 存储中间处理图像
                std::vector<cv::Mat> intermediate_imgs;
                intermediate_imgs.push_back(crop.clone()); // 原始裁剪图像

                // 图像处理
                cv::Mat gray, gray_inv, binary;
                if (crop.channels() == 3)
                    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
                else
                    gray = crop;
                intermediate_imgs.push_back(gray.clone()); // 灰度图像
                // 反色处理
                cv::bitwise_not(gray, gray_inv);
                intermediate_imgs.push_back(gray_inv.clone());

                // 伽马校正
                cv::Mat lut(1, 256, CV_8U);
                uchar* p = lut.ptr();
                float inv_gamma = 1.2;
                for (int i = 0; i < 256; ++i) p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, inv_gamma) * 255.0);
                cv::LUT(gray_inv, lut, gray_inv);
                intermediate_imgs.push_back(gray_inv.clone()); // 伽马校正后

                //int min_dim = std::min(gray_inv.rows, gray_inv.cols); // 57
                //int blockSize = std::max(3, min_dim / 15); // 57/15≈3.8→3
                //if (blockSize % 2 == 0) blockSize++; // 确保奇数→3

                //// 自适应阈值二值化
                //cv::adaptiveThreshold(gray_inv, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize, -5);
                //double thresh = cv::threshold(gray_inv, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
                binary = selectThresholdAdaptive(gray_inv);
                intermediate_imgs.push_back(binary.clone()); // 二值化结果

                // 闭运算
                cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
                intermediate_imgs.push_back(binary.clone());

                // 轮廓提取
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                // 剔除面积小于等于42的轮廓，保留面积大于42的轮廓
                contours.erase(std::remove_if(contours.begin(), contours.end(),
                    [](const std::vector<cv::Point>& contour) {
                        return cv::contourArea(contour) <= 42;  // 剔除条件：面积<=42
                    }), contours.end());
                //// 创建新向量只保留面积大于42的轮廓
                //std::vector<std::vector<cv::Point>> filtered_contours;
                //std::copy_if(contours.begin(), contours.end(), std::back_inserter(filtered_contours),
                //    [](const std::vector<cv::Point>& contour) {
                //        return cv::contourArea(contour) > 42;  // 保留条件：面积>42
                //    });
                //contours = std::move(filtered_contours);  // 替换原轮廓向量
                cv::Mat contour_img_transparent = cv::Mat::zeros(crop.size(), CV_8UC3);
                for (const auto& contour : contours) {
                    std::vector<std::vector<cv::Point>> single_contour = { contour };
                    cv::drawContours(contour_img_transparent, single_contour, -1,
                        cv::Scalar(0, 255, 0), 1, cv::LINE_AA); // 使用抗锯齿
                }

                // 合并到原图
                cv::Mat contour_img;
                if (crop.channels() == 3) {
                    contour_img = crop.clone(); // 已经是 BGR
                }
                else if (crop.channels() == 1) {
                    cv::cvtColor(crop, contour_img, cv::COLOR_GRAY2BGR);
                }
                else if (crop.channels() == 4) {
                    cv::cvtColor(crop, contour_img, cv::COLOR_BGRA2BGR);
                }
                else {
                    throw std::runtime_error("Unsupported channel count in crop");
                }
                cv::addWeighted(contour_img, 1.0, contour_img_transparent, 0.5, 0, contour_img);
                intermediate_imgs.push_back(contour_img); // 轮廓图像

                // 计算轮廓总面积
                double area_contour = 0.0;
                for (const auto& cnt : contours) {
                    double a = cv::contourArea(cnt);
                    if (a > 20) area_contour += a; // 过滤小面积
                }

                // 确保轮廓总面积不超过原区域面积
                if (area_contour > res.area) area_contour = res.area;
                res.contours = contours;
                res.area_contour = static_cast<float>(area_contour);

                // 横向拼接所有中间图像
                //if (!intermediate_imgs.empty()) {
                if(true){
                    int total_width = 0;
                    int max_height = 0;

                    // 计算总宽度和最大高度
                    for (const auto& img : intermediate_imgs) {
                        total_width += img.cols;
                        if (img.rows > max_height) {
                            max_height = img.rows;
                        }
                    }

                    // 创建拼接画布
                    cv::Mat combined_img(max_height, total_width, intermediate_imgs[0].type(), cv::Scalar(0));

                    // 拼接图像
                    int current_x = 0;
                    for (const auto& img : intermediate_imgs) {
                        cv::Mat roi = combined_img(cv::Rect(current_x, 0, img.cols, img.rows));

                        // 如果是单通道图像，转换为三通道显示
                        if (img.channels() == 1) {
                            cv::cvtColor(img, roi, cv::COLOR_GRAY2BGR);
                        }
                        else {
                            img.copyTo(roi);
                        }

                        current_x += img.cols;
                    }
                    //std::cout << "\t拼接图像已生成: " << std::endl;
                    // 保存拼接图像
                    size_t last_backslash = image_path.find_last_of('\\');
                    size_t last_dot = image_path.find_last_of('.');

                    if (last_backslash != std::string::npos) {
                        // 构建 DK_process 目录和完整保存路径
                        fs::path src_path(image_path);
                        //std::cout << "src_path" << src_path << std::endl;
                        fs::path parent_dir = src_path.parent_path();
                        //std::cout << "parent_dir" << parent_dir << std::endl;
                        fs::path dk_process_dir = parent_dir / "DK_process";
                        //std::cout << "dk_process_dir" << dk_process_dir << std::endl;
                        std::string filename = src_path.stem().string(); // 只取文件名，不带扩展名
                        std::string extension = src_path.extension().string(); // 带点的扩展名
                        id++;
                        fs::path save_path = dk_process_dir / (filename + "_process" + std::to_string(id) + "." + extension);

                        // 递归创建所有父目录
                        fs::create_directories(save_path.parent_path());

                        if (combined_img.empty()) {
                            std::cerr << "拼接图像为空，无法保存！" << std::endl;
                        }
                        // 保存到 DK_process 路径
                        bool success = cv::imwrite(save_path.string(), combined_img);
                        //if (success) {
                        //    std::cout << "图像已保存到: " << save_path << std::endl;
                        //}
                        //else {
                        //    std::cerr << "保存失败: " << save_path << std::endl;
                        //}
                    }
                }

            }
            catch (const cv::Exception& e) {
                std::cerr << "OpenCV 异常: " << e.what()
                    << " 在文件: " << e.file
                    << " 行: " << e.line << std::endl;
            }
        }

        results.push_back(res);
    }
    return results;
}

INCEPTIONDLL_API std::vector<DetectionResult> YOLO12Infer::infer(const std::string& image_path) {
    cv::Mat original_img;
    float h_ratio, w_ratio;
    std::vector<float> input_tensor = preprocess(image_path, original_img, h_ratio, w_ratio);

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

    // 运行模型
    auto output_tensors = session_.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor_ort, 1, output_names.data(), 1);
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();

    std::vector<int64_t> output_shape = type_info.GetShape();

    // 打印原始输出形状
    //if (TestModel_Flag) {
    //    std::cout << "原始模型输出形状: [";
    //    for (size_t i = 0; i < output_shape.size(); ++i) {
    //        std::cout << output_shape[i] << (i < output_shape.size() - 1 ? ", " : "");
    //    }
    //    std::cout << "]" << std::endl;
    //}

    // 假设原始输出是[1, 10, 5376]或[1, 10, 5376, 1]，需要转置为[5376, 10]
    int orig_rows = static_cast<int>(output_shape[1]);  // 10
    int orig_cols = static_cast<int>(output_shape[2]);  // 5376

    std::vector<float> transposed_output(orig_cols * orig_rows);
    for (int i = 0; i < orig_rows; ++i) {
        for (int j = 0; j < orig_cols; ++j) {
            // 转置: [i, j] -> [j, i]
            transposed_output[j * orig_rows + i] = output_data[i * orig_cols + j];
        }
    }
    // 打印转置后形状
    //if (TestModel_Flag) {
    //    std::cout << "转置后模型输出形状: [" << orig_cols << ", " << orig_rows << "]" << std::endl;

    //    // 打印前几行，用于与Python对比
    //    std::cout << "转置后的前3行数据：" << std::endl;
    //    for (int i = 0; i < std::min(3, orig_cols); ++i) {
    //        std::cout << "[";
    //        for (int j = 0; j < orig_rows; ++j) {
    //            std::cout << transposed_output[i * orig_rows + j];
    //            if (j < orig_rows - 1) std::cout << ", ";
    //        }
    //        std::cout << "]" << std::endl;
    //    }
    //}

    return postprocess(transposed_output, orig_cols, orig_rows, h_ratio, w_ratio, original_img, image_path);
}

INCEPTIONDLL_API void YOLO12Infer::draw_box(cv::Mat& img, const DetectionResult& res, bool show_score, bool show_class) {


    //=====全部标注======
    //     std::string label;
    //    auto it = InceptionDLL::CLASS_COLORS.find(res.class_name);
    //      cv::Scalar color = (it != InceptionDLL::CLASS_COLORS.end()) ? it->second : cv::Scalar(0, 255, 255);
    //      cv::rectangle(img, res.bbox, color, 1);
    //     if (show_class) label += res.class_name;
    //     if (show_score) label += (label.empty() ? "" : " ") + std::to_string(res.confidence).substr(0, 4);
    //if (!label.empty()) {
    //    int baseLine = 0;
    //    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    //    cv::rectangle(img, cv::Rect(res.bbox.x, res.bbox.y - label_size.height - 4, label_size.width, label_size.height + 4), color, -1);
    //    cv::putText(img, label, cv::Point(res.bbox.x, res.bbox.y - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    //}

    //=======只对DK类和CS类及WZ类进行标注========
    if (!(res.class_name == "DK_A" || res.class_name == "DK_B" || res.class_name == "DK_C" ||
          res.class_name == "CS_A" || res.class_name == "CS_B" || res.class_name == "CS_C" || 
          res.class_name == "WZ"))
    {
        return;
    }
    // 修改 label，根据前缀统一成 DK 或 CS
    std::string label;
    if (res.class_name.find("DK") != std::string::npos) {
        label = "DK";
    }
    else if (res.class_name.find("CS") != std::string::npos) {
        label = "CS";
    }
    else if (res.class_name == "WZ") {
        label = "WZ";
    }

    if (show_score) {
        label += " " + std::to_string(res.confidence).substr(0, 4);
    }

    auto it = InceptionDLL::CLASS_COLORS.find(res.class_name);
    cv::Scalar color = (it != InceptionDLL::CLASS_COLORS.end()) ? it->second : cv::Scalar(0, 255, 255);

    // 绘制矩形
    cv::rectangle(img, res.bbox, color, 1.5);

    if (!label.empty()) {
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        if (label_size.height < 2 * res.bbox.height) {
            cv::rectangle(img, cv::Rect(res.bbox.x - label_size.width, res.bbox.y, label_size.width - 5, label_size.height), color, -1);
            cv::putText(img, label, cv::Point(res.bbox.x - label_size.width + 3, res.bbox.y + 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
            if (InceptionDLL::chinese_labels.find(res.class_name) != InceptionDLL::chinese_labels.end()) {
                cv::Mat label_img = InceptionDLL::chinese_labels[res.class_name];
                cv::Rect roi(res.bbox.x, res.bbox.y - label_img.rows, label_img.cols, label_img.rows);
                label_img.copyTo(img(roi));
            }
        }
        else {
            cv::rectangle(img, cv::Rect(res.bbox.x - label_size.width, res.bbox.y +label_size.height+ 40, label_size.width - 5, label_size.height), color, -1);
            cv::putText(img, label, cv::Point(res.bbox.x - label_size.width + 3, res.bbox.y + label_size.height + 50), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
            if (InceptionDLL::chinese_labels.find(res.class_name) != InceptionDLL::chinese_labels.end()) {
                cv::Mat label_img = InceptionDLL::chinese_labels[res.class_name];
                cv::Rect roi(res.bbox.x, res.bbox.y - label_img.rows, label_img.cols, label_img.rows);
                label_img.copyTo(img(roi));
            }
        }
    }

    // 画轮廓（红色）
    if (!res.contours.empty() && res.class_name.find("DK") != std::string::npos) {
        for (const auto& cnt : res.contours) {
            double area = cv::contourArea(cnt);
            if (area < 50.0) {
                continue; // 跳过小轮廓
            }
            std::vector<cv::Point> cnt_shifted;
            for (const auto& pt : cnt) {
                cnt_shifted.emplace_back(pt.x + res.bbox.x, pt.y + res.bbox.y);
            }
            cv::drawContours(img, std::vector<std::vector<cv::Point>>{cnt_shifted}, -1, cv::Scalar(0, 0, 255), 1);
        }
    }
}

INCEPTIONDLL_API std::string YOLO12Infer::predict(const std::string& image_path, bool visual, bool show_score, bool show_class, bool save_or_not) {
    std::vector<DetectionResult> results = infer(image_path);
    cv::Mat img = InceptionUtils::imread_unicode(image_path, cv::IMREAD_COLOR);

    // 过滤掉面积小于100的检测结果
    std::vector<DetectionResult> filtered_results;
    for (const auto& res : results) {
        if (res.area > 100) {
            filtered_results.push_back(res);
        }
        else if (TestModel_Flag) {
            std::cout << "过滤掉小面积检测框: " << res.class_name << ", 面积=" << res.area << std::endl;
        }
    }
    results = std::move(filtered_results);
    if (results.empty()) {
        return R"([{"class_name": "ZC"}])";
    }
    if (Data_Collector) {

        bool has_cs_c = false;
        for (const auto& res : results) {
            if (res.class_name == "CS_C") {
                has_cs_c = true;
                break;
            }
        }
        // 只有当含有 CS_C 类时才保存
        if (has_cs_c) {
            fs::path p(image_path);

            // 获取根目录 (D:)
            fs::path root_path = p.root_path();

            // 构建目标目录
            std::string data_collector_dir = root_path.string() + "Data_Collector";
            fs::create_directories(data_collector_dir);

            // 从路径中提取日期时间信息 (WP4_2025Y08M04D07h03m48s)
            std::string datetime_info = p.parent_path().parent_path().filename().string();

            // 确定相机类型 (右相机 -> R, 左相机 -> L)
            std::string camera_type = "U"; // 默认为未知类型
            std::string camera_folder = p.parent_path().filename().string();
            if (camera_folder == "右相机_railhead_stretch") {
                camera_type = "R";
            }
            else if (camera_folder == "左相机_railhead_stretch") {
                camera_type = "L";
            }

            // 获取原始文件名
            std::string original_filename = p.filename().string();

            // 构建新的文件名
            std::string new_filename = datetime_info + "_" + camera_type + "_" + original_filename;

            // 构建完整的目标路径
            std::string dest_path = data_collector_dir + "//" + new_filename;
            if (!img.empty()) {
                InceptionUtils::imwrite_unicode(dest_path, img);
                if (TestModel_Flag) {
                    std::cout << "发现CS_C类，已保存图像到: " << dest_path << std::endl;
                }
            }
        }
    }
    //
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
        std::string window_name = "YOLO12_Detection_" + img_name;
        // 使用更可靠的方式显示和等待
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::imshow(window_name, img);

        cv::waitKey(0); // 等待按键
        cv::destroyAllWindows(); // 销毁所有窗口而不是特定窗口
    }
    return detection_results_to_string(results);
}

