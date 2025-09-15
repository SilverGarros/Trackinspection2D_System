// InceptionDLL.cpp : 定义 DLL 的导出函数。

#include "pch.h"
#include "framework.h"
#include "InceptionDLL_v0.3.0.h"
#include "InceptionUtils_v0.3.0.h"
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <regex>

bool TestModel_Flag = false;
bool Data_Collector = false;
bool Save_Single_Defects = true; // Add 0.2.9 按条缺陷标记保存控制器开关
bool Save_Defects_with_Combiner = true;// Add 0.3.0 合并缺陷标记保存控制器开关

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
        obj["id"] = res.id;
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
    const std::unordered_map<int, std::string> classes_lable_map_for_v3 = {
        {0, "YC"},
        {1, "DKCS"},
        {2, "BM"},
        {3, "HF"},
        {4, "ZC"},
        {5, "GF"},
        {6, "ZHC"},
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
    // Update InceptionDLL_v0.3.0 轨面提取优化
    INCEPTIONDLL_API cv::Mat CropRailhead(const std::string& img_path, int& offset_x, int crop_threshold, int crop_kernel_size, int crop_wide, bool center_limit, int limit_area) {
        cv::Mat img = imread_unicode(img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "CropRailhead: 图像读取失败: " << img_path << std::endl;
            return cv::Mat();
        }
        return RailheadCropHighlightCenterArea(img, offset_x, crop_threshold,
            crop_kernel_size, crop_wide,
            center_limit, limit_area);
    }
    //Update InceptionDLL_v0.3.0_N+2 轨面提取优化（添加轨面提取的x轴偏移量&Fix裁切图像宽度小于设置宽度的情况）  
    INCEPTIONDLL_API cv::Mat RailheadCropHighlightCenterArea(
        const cv::Mat& img, int& offset_x, int threshold, int kernel_size, int crop_wide, bool center_limit = true, int limit_area)
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
        //Fix 裁切图像宽度小于设置宽度的情况
        if (x2 - x1 != crop_wide) {
            x2 = x1 + crop_wide;
            if (x2 > img.cols) {
                x2 = img.cols;
                x1 = x2 - crop_wide;
                x1 = std::max(0, x1);
            }
        }

        // 设置offset_x为裁剪区域左上角的x坐标
        offset_x = x1;

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
    //INCEPTIONDLL_API cv::Mat RailheadCropHighlightCenterArea(
    //    const cv::Mat& img,
    //    int threshold,
    //    int kernel_size,
    //    int crop_wide,
    //    bool center_limit,
    //    int limit_area=50,
    //    int center_axis_x= 300) {

    //    center_axis_x = -1;
    //    // 确保crop_wide不超过图像宽度
    //    crop_wide = std::min(crop_wide, img.cols);

    //    cv::Mat img_gray;
    //    if (img.channels() == 3)
    //        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    //    else
    //        img_gray = img.clone();

    //    cv::Mat binary;
    //    cv::threshold(img_gray, binary, threshold, 255, cv::THRESH_BINARY);

    //    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    //    cv::Mat closed;
    //    cv::morphologyEx(binary, closed, cv::MORPH_CLOSE, kernel);

    //    std::vector<std::vector<cv::Point>> contours;
    //    cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


    //    int img_center = img.cols / 2;
    //    int crod_m = img_center;
    //    if (!contours.empty()) {
    //        auto largest = std::max_element(contours.begin(), contours.end(),
    //            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
    //                return cv::contourArea(a) < cv::contourArea(b);
    //            });
    //        cv::Rect bbox = cv::boundingRect(*largest);
    //        crod_m = bbox.x + bbox.width / 2;
    //        if (center_limit) {
    //            if (std::abs(crod_m - img_center) > limit_area)
    //                crod_m = img_center;
    //        }
    //    }
    //    // 调整中轴位置，确保裁剪宽度等于crop_wide
    //    int half_width = crop_wide / 2;

    //    // 如果中轴太靠左，向右调整
    //    if (crod_m < half_width) {
    //        crod_m = half_width;
    //    }
    //    // 如果中轴太靠右，向左调整
    //    else if (crod_m > img.cols - half_width) {
    //        crod_m = img.cols - half_width;
    //    }
    //    center_axis_x = crod_m;


    //    int x1 = std::max(0, crod_m - half_width);
    //    int x2 = std::min(img.cols, crod_m + half_width);
    //    int y1 = 0, y2 = img.rows;
    //    return img(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    //}

    // Addv0.3.0_N+1 添加分类置信度
    INCEPTIONDLL_API ClassificationResult ClassPredictOnnx(
        Ort::Session& session, const cv::Mat& img_input, int img_size)
    {
        cv::Mat img;
        if (img_input.empty()) {
            std::cout << "ClassPredictOnnx 获取的图像为空" << std::endl;
            //return "110 Unknown";
            return { "110 Unknown", 0.0f, -1 }; // 返回错误结构体
        }
        cv::resize(img_input, img, cv::Size(img_size, img_size));
        img.convertTo(img, CV_32F, 1.0 / 255);
        std::vector<float> input_tensor_values;
        input_tensor_values.reserve(img_size * img_size * 3);
        cv::Mat channels[3];
        cv::split(img, channels);
        for (int c = 2; c >= 0; c--) {// BGR -> RGB
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
            float max_value = output_data[0];
            int cur_index = 0;
            float cur_value = output_data[0];

            std::vector<float> probabilities(output_count);
            // 1. 先找到最大值（数值稳定性技巧）
            float max_logit = output_data[0];
            for (size_t i = 1; i < output_count; i++) {
                if (output_data[i] > max_logit) {
                    max_logit = output_data[i];
                }
            }
        
            // 2. 计算softmax并同时查找最大概率
            float sum_exp = 0.0f;
            float max_prob = 0.0f;
            int max_index = 0;

            for (size_t i = 0; i < output_count; i++) {
                probabilities[i] = std::exp(output_data[i] - max_logit); // 减去最大值提高数值稳定性
                sum_exp += probabilities[i];
            }
        
            // 3. 归一化得到概率
            for (size_t i = 0; i < output_count; i++) {
                probabilities[i] /= sum_exp;
                // 剔除 ~i=2BM~ i=3 HF和i=6GF ~i=7GD~ 的影响
                //if (i == 3 || i == 6) {
                //    continue;
                //}
                //if (i == 3) {
                //    continue;
                //}

                // 查找最大概率的索引
                if (probabilities[i] > max_prob) {
                    max_prob = probabilities[i];
                    max_index = static_cast<int>(i);
                }
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
                    /*
                    {0, "YC"},
                    {1, "DKCS"},
                    {2, "BM"},
                    {3, "HF"},
                    {4, "ZC"},
                    {5, "GF"},
                    {6, "ZHC"},
                    {7, "GD"}
                    */
            }
            // 添加置信度过滤逻辑
            if (TestModel_Flag) {
                std::cout << "=== 所有类别置信度 ===" << std::endl;
                std::vector<std::string> labels2 = { "YC", "DK", "BM", "HF", "CS", "ZC", "GF", "GD" };
                std::vector<std::string> labels1 = { "YC", "DKCS", "BM", "HF", "ZC", "GF", "ZHC", "GD" };
                // 打印第二个映射的置信度
                for (size_t i = 0; i < output_count && i < labels2.size(); i++) {
                    std::cout << labels2[i] << ": " << std::fixed << std::setprecision(4)
                        << probabilities[i] * 100 << "%";
                    if (i == max_index) {
                        std::cout << " <- 最高";
                    }
                    std::cout << std::endl;
                }
                std::cout << "=====================" << std::endl;
                std::cout << "最高置信度: " << std::fixed << std::setprecision(4)
                    << max_prob * 100 << "% (索引: " << max_index << ")" << std::endl;
            }

            float confidence_threshold = 0.4; // 根据实际情况调整
            if (max_index == 2) {
                confidence_threshold = 0.8; // BM类别的阈值更高
            }
            if (max_prob < confidence_threshold) {
                return { "ZC", max_prob, max_index };  // 置信度不足，返回默认类别
            }
            auto it = InceptionDLL::classes_lable_map.find(max_index);
            //return (it != InceptionDLL::classes_lable_map.end()) ? it->second : "111 Unknown";
            std::string label = (it != InceptionDLL::classes_lable_map.end()) ? it->second : "111 Unknown";
            // 返回结构体
            return { label, max_prob, max_index };
        }
        catch (...) {
            return { "112 Unknown", 0.0f, -1 };
        }
    }

    INCEPTIONDLL_API ClassificationResult ClassifyImage(
        Ort::Session& classify_session,
        const std::string& img_path,
        int img_size,
        const std::string& temp_path)
    {
        cv::Mat img = imread_unicode(img_path);
        if (img.empty()) {
            std::cout << "ClassifyImage 获取的图像为空: " << img_path << std::endl;
            return { "110 Unknown", 0.0f, -1 };
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
                std::string cls;
                for (const auto& item : json_results) {
                    cls = item.value("class_name", "");
                    //if (cls == "DK_A" || cls == "DK_B" || cls == "DK_C" ||
                    //    cls == "CS_A" || cls == "CS_B" || cls == "CS_C" || cls == "WZ")
                    // Update 0.2.9 以适应DK_ADK1
                    std::regex pattern("(DK_[ABC]|CS_[ABC]|WZ)");
                    if (std::regex_search(cls, pattern)) {
                        containsDesired = true;
                        break;
                    }
                }
                //std::cout << "===临时DEBUG===：解析到的 JSON CLASS为:" << cls << std::endl;
                if (!containsDesired) {
                    //std::cout << "===临时DEBUG===：JSON 数组结果被修正为ZC" << cls << std::endl;
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
                    std::cout << "执行掉块擦伤类处理....." << std::endl;
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

    //Add 0.2.9 定位结果按照→↓顺序排序
    // 按照阅读顺序排列（从上到下，从左到右）
    std::sort(results.begin(), results.end(), [](const DetectionResult& a, const DetectionResult& b) {
        // 如果两个框在同一行（Y坐标相差不大），按X坐标排序
        if (std::abs(a.bbox.y - b.bbox.y) < 20) { // 20像素内的认为是同一行
            return a.bbox.x < b.bbox.x;
        }
        // 否则按Y坐标排序
        return a.bbox.y < b.bbox.y;
        });
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

INCEPTIONDLL_API void YOLO12Infer::save_single_defect_image(const cv::Mat& img, const DetectionResult& res,
    const std::string& original_path, const std::string& label_with_id) {
    try {
        cv::Mat defect_img = img.clone();

        // 构建保存路径
        fs::path p(original_path);
        std::string save_dir = p.parent_path().string() + "//detection_result";
        fs::create_directories(save_dir);

        // 生成文件名：原文件名_类别ID.扩展名
        std::string stem = p.stem().string();
        std::string extension = p.extension().string();
        std::string save_file = save_dir + "//" + stem + "_" + label_with_id + extension;

        InceptionUtils::imwrite_unicode(save_file, defect_img);

    }
    catch (const std::exception& e) {
        std::cerr << "Failed to save single defect image: " << e.what() << std::endl;
    }
}
INCEPTIONDLL_API void YOLO12Infer::draw_box(cv::Mat& img, DetectionResult& res, bool show_score, bool show_class,std::map<std::string, int>& class_counter, bool save_single_defects,const std::string& original_image_path) {
    cv::Mat clean_img = img.clone();
    //=======只对DK类和CS类及WZ类进行标注========
    if (!(res.class_name == "DK_A" || res.class_name == "DK_B" || res.class_name == "DK_C" || res.class_name == "DK_AB" || res.class_name == "DK"||
          res.class_name == "CS_A" || res.class_name == "CS_B" || res.class_name == "CS_C" || res.class_name == "CS_AB" || res.class_name == "CS" ||
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
    // 获取并递增类别计数
    int class_id = ++class_counter[label];
    std::string label_with_id = label + std::to_string(class_id);
    std::string save_img_name_info = label_with_id;

    if (show_score) {
        label_with_id += " " + std::to_string(res.confidence).substr(0, 4);
    }
    // 颜色映射
    auto it = InceptionDLL::CLASS_COLORS.find(res.class_name);
    cv::Scalar color = (it != InceptionDLL::CLASS_COLORS.end()) ? it->second : cv::Scalar(0, 255, 255);
    // 绘制矩形
    cv::rectangle(clean_img, res.bbox, color, 1.5);
    if (!label.empty()) {
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        if (label_size.height < 2 * res.bbox.height) {
            cv::rectangle(clean_img, cv::Rect(res.bbox.x - label_size.width, res.bbox.y, label_size.width - 5, label_size.height), color, -1);
            cv::putText(clean_img, label, cv::Point(res.bbox.x - label_size.width + 3, res.bbox.y + 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
            if (InceptionDLL::chinese_labels.find(res.class_name) != InceptionDLL::chinese_labels.end()) {
                cv::Mat label_img = InceptionDLL::chinese_labels[res.class_name];
                cv::Rect roi(res.bbox.x, res.bbox.y - label_img.rows, label_img.cols, label_img.rows);
                label_img.copyTo(clean_img(roi));
            }
        }
        else {
            cv::rectangle(img, cv::Rect(res.bbox.x - label_size.width, res.bbox.y + label_size.height + 40, label_size.width - 5, label_size.height), color, -1);
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
            cv::drawContours(clean_img, std::vector<std::vector<cv::Point>>{cnt_shifted}, -1, cv::Scalar(0, 0, 255), 1);
        }
    }
    // 单独保存缺陷图片
    if (save_single_defects && !original_image_path.empty()) {
        save_single_defect_image(clean_img, res, original_image_path, save_img_name_info);
        //// 生成文件名：原文件名_类别ID.扩展名
        //fs::path p(original_image_path);
        //std::string stem = p.stem().string();
        //std::string extension = p.extension().string();
        //std::string save_file = stem + "_" + original_image_path + extension;
        res.class_name += "-" + save_img_name_info;
    }

}
//Add InceptionDLL_v0.3.0_N+4 添加`merge_detection_results_by_class`定义实现
INCEPTIONDLL_API std::vector<DetectionResult>  merge_detection_results_by_class(
    const std::vector<DetectionResult>& results) {

    std::vector<DetectionResult> merged_results;

    if (results.empty()) {
        return merged_results;
    }

    // 按合并后的类别名称分组
    std::map<std::string, std::vector<DetectionResult>> grouped_results;

    for (const auto& res : results) {
        std::string merged_class_name = res.class_name;

        // 应用合并规则
        if (res.class_name == "DK_A" || res.class_name == "DK_B") {
            merged_class_name = "DK_AB";
        }
        else if (res.class_name == "CS_A" || res.class_name == "CS_B") {
            merged_class_name = "CS_AB";
        }
        // DK_C, CS_C, WZ, HF, GF 保持原样，不合并

        grouped_results[merged_class_name].push_back(res);
    }
    // 对每个组进行合并处理
    for (auto& group : grouped_results) {
        std::string group_name = group.first;
        std::vector<DetectionResult>& group_results = group.second;

        if (group_results.empty()) continue;

        // 判断是否需要合并
        bool should_merge = group_results.size() > 1;
        if (should_merge) {
            // 合并该组的所有检测结果
            DetectionResult merged;
            merged.class_name = group_name;
            merged.id = 0;
            merged.confidence = 0;

            // 初始化合并的bbox为第一个结果的bbox
            merged.bbox = group_results[0].bbox;
            // 合并所有bbox（取并集）
            for (size_t i = 1; i < group_results.size(); ++i) {
                merged.bbox |= group_results[i].bbox;
            }

            // 合并置信度（取平均值）
            float total_confidence = 0;
            for (const auto& res : group_results) {
                total_confidence += res.confidence;
            }
            merged.confidence = total_confidence / group_results.size();
            // 计算合并后的面积
            merged.area = merged.bbox.area();
            // 合并面积轮廓面积
            merged.area_contour = 0;
            for (const auto& res : group_results) {
                merged.area_contour += res.area_contour;
            }
            // 合并轮廓（需要调整坐标到新的合并bbox内）
            for (const auto& res : group_results) {
                for (const auto& cnt : res.contours) {
                    std::vector<cv::Point> adjusted_cnt;
                    for (const auto& pt : cnt) {
                        // 调整坐标：从原始bbox内坐标转换为合并bbox内坐标
                        int adjusted_x = pt.x + res.bbox.x - merged.bbox.x;
                        int adjusted_y = pt.y + res.bbox.y - merged.bbox.y;
                        adjusted_cnt.emplace_back(adjusted_x, adjusted_y);
                    }
                    if (!adjusted_cnt.empty()) {
                        merged.contours.push_back(adjusted_cnt);
                    }
                }
            }
            // 记录合并信息
            if (TestModel_Flag && group_results.size() > 1) {
                std::cout << "合并 " << group_name << " 类: "
                    << group_results.size() << " 个检测框 -> 1 个合并框" << std::endl;
                std::cout << "  合并后坐标: (" << merged.bbox.x << ", " << merged.bbox.y
                    << ", " << merged.bbox.width << ", " << merged.bbox.height << ")" << std::endl;
                std::cout << "  合并后面积: " << merged.area << std::endl;
            }

            merged_results.push_back(merged);
        }
        else {
            // 不合并，直接添加所有结果（但更新类别名称为分组名称）
            for (const auto& det : group_results) {
                DetectionResult result = det;
                // 只有当分组名称与原始名称不同时才更新
                if (result.class_name != group_name) {
                    result.class_name = group_name;
                }
                merged_results.push_back(result);
            }
        }
    }
    // 按类别名称排序，确保输出顺序一致
    std::sort(merged_results.begin(), merged_results.end(),
        [](const DetectionResult& a, const DetectionResult& b) {
            return a.class_name < b.class_name;
        });

    return merged_results;
};

//// 新增辅助函数：绘制单个检测框
//INCEPTIONDLL_API void YOLO12Infer::draw_box_For_draw_box_classes(cv::Mat& img, DetectionResult& res,
//    bool show_score, bool show_class, std::map<std::string, int>& class_counter) {
//
//    std::string label;
//    if (res.class_name.find("DK") != std::string::npos) {
//        label = "DK";
//    }
//    else if (res.class_name.find("CS") != std::string::npos) {
//        label = "CS";
//    }
//    else if (res.class_name == "WZ") {
//        label = "WZ";
//    }
//
//    int class_id = ++class_counter[label];
//    std::string label_with_id = label + std::to_string(class_id);
//
//    if (show_score) {
//        label_with_id += " " + std::to_string(res.confidence).substr(0, 4);
//    }
//
//    // 颜色映射
//    auto it = InceptionDLL::CLASS_COLORS.find(res.class_name);
//    cv::Scalar color = (it != InceptionDLL::CLASS_COLORS.end()) ?
//        it->second : cv::Scalar(0, 255, 255);
//
//    // 绘制矩形
//    cv::rectangle(img, res.bbox, color, 1.5);
//
//    if (show_class && !label.empty()) {
//        int baseLine = 0;
//        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//        if (label_size.height < 2 * res.bbox.height) {
//            cv::rectangle(img, cv::Rect(res.bbox.x - label_size.width, res.bbox.y,
//                label_size.width - 5, label_size.height), color, -1);
//            cv::putText(img, label, cv::Point(res.bbox.x - label_size.width + 3,
//                res.bbox.y + 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
//        }
//    }
//}
////Add InceptionDLL_v0.3.0_N+4 添加`draw_box_classes`定义实现
//INCEPTIONDLL_API void YOLO12Infer::draw_box_classes(cv::Mat& img, std::vector<DetectionResult>& results,
//    bool show_score, bool show_class, std::map<std::string, int>& class_counter,
//    bool save_single_defects, const std::string& original_image_path) {
//    //=======只对DK类和CS类及WZ类进行标注========
//    std::vector<DetectionResult> filtered_results;
//    for (const auto& res : results) {
//        if (res.class_name == "DK_A" || res.class_name == "DK_B" || res.class_name == "DK_C" ||
//            res.class_name == "CS_A" || res.class_name == "CS_B" || res.class_name == "CS_C" ||
//            res.class_name == "WZ") {
//            filtered_results.push_back(res);
//        }
//    }
//
//    // 按类别分组（DK_C和CS_C单独处理）
//    std::map<std::string, std::vector<DetectionResult>> grouped_results;
//    for (const auto& res : filtered_results) {
//        // DK_C和CS_C单独作为一类，不合并
//        if (res.class_name == "DK_C" || res.class_name == "CS_C") {
//            grouped_results[res.class_name].push_back(res);
//        }
//        // DK_A和DK_B合并为DK_AB
//        else if (res.class_name == "DK_A" || res.class_name == "DK_B") {
//            grouped_results["DK_AB"].push_back(res);
//        }
//        // CS_A和CS_B合并为CS_AB
//        else if (res.class_name == "CS_A" || res.class_name == "CS_B") {
//            grouped_results["CS_AB"].push_back(res);
//        }
//        // WZ单独作为一类
//        else if (res.class_name == "WZ") {
//            grouped_results["WZ"].push_back(res);
//        }
//        else 
//        {
//            grouped_results[res.class_name].push_back(res);
//        }
//    }
//    for (auto& group : grouped_results) {
//        std::string group_name = group.first;
//        std::vector<DetectionResult>& group_results = group.second;
//        if (group_results.empty()) continue;
//
//        // 如果只有一个结果或不需要合并（DK_C、CS_C），直接绘制
//        if (group_results.size() == 1 ||
//            group_name == "DK_C" || group_name == "CS_C") {
//            // 直接绘制单个结果
//            DetectionResult& res = group_results[0];
//            cv::Mat clean_img = img.clone();
//            draw_box(clean_img, res, show_score, show_class);
//            // 画轮廓（红色）- 仅DK类
//            if (!res.contours.empty() && res.class_name.find("DK") != std::string::npos) {
//                for (const auto& cnt : res.contours) {
//                    double area = cv::contourArea(cnt);
//                    if (area < 50.0) continue;
//
//                    std::vector<cv::Point> cnt_shifted;
//                    for (const auto& pt : cnt) {
//                        cnt_shifted.emplace_back(pt.x + res.bbox.x, pt.y + res.bbox.y);
//                    }
//                    cv::drawContours(clean_img, std::vector<std::vector<cv::Point>>{cnt_shifted},
//                        -1, cv::Scalar(0, 0, 255), 1);
//                }
//            }
//
//            if (save_single_defects && !original_image_path.empty()) {
//                // 获取类别ID
//                std::string label;
//                if (res.class_name.find("DK") != std::string::npos) {
//                    label = "DK";
//                }
//                else if (res.class_name.find("CS") != std::string::npos) {
//                    label = "CS";
//                }
//                else if (res.class_name == "WZ") {
//                    label = "WZ";
//                }
//                int class_id = ++class_counter[label];
//                std::string label_with_id = label + std::to_string(class_id);
//
//                save_single_defect_image(clean_img, res, original_image_path, label_with_id);
//                res.class_name += "-" + label_with_id;
//            }
//
//        }
//        else {
//            // 合并多个结果（用于DK_AB、CS_AB）
//            DetectionResult merged_res = merge_detection_results_by_class(group_results);
//            // 更新类别名称
//            merged_res.class_name = group_name;
//            cv::Mat clean_img = img.clone();
//            draw_box_For_draw_box_classes(clean_img, merged_res, show_score, show_class, class_counter);
//
//            // 画合并后的轮廓
//            if (!merged_res.contours.empty() && group_name.find("DK") != std::string::npos) {
//                for (const auto& cnt : merged_res.contours) {
//                    cv::drawContours(clean_img, std::vector<std::vector<cv::Point>>{cnt},
//                        -1, cv::Scalar(0, 0, 255), 1);
//                }
//            }
//            if (save_single_defects && !original_image_path.empty()) {
//                std::string label;
//                if (group_name.find("DK") != std::string::npos) {
//                    label = "DK";
//                }
//                else if (group_name.find("CS") != std::string::npos) {
//                    label = "CS";
//                }
//                int class_id = ++class_counter[label];
//                std::string label_with_id = label + std::to_string(class_id);
//
//                save_single_defect_image(clean_img, merged_res, original_image_path, label_with_id);
//                // 更新原始结果中的第一个结果的类别名称
//                group_results[0].class_name = group_name + "-" + label_with_id;
//            }
//
//        }
//
//    }
//
//}
//Update InceptionDLL_v0.3.0_N+4 修改predict 添加按类别合并标注
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

    //==== 内置数据搜集器，有需要再打开 =====//
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
            if (camera_folder == "右相机_railhead_stretch"|| camera_folder == "DATR_railhead_stretch") {
                camera_type = "R";
            }
            else if (camera_folder == "左相机_railhead_stretch" || camera_folder == "DATL_railhead_stretch") {
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
    // Add 0.2.9 加类别计数器 //
    std::map<std::string, int> class_counter = { {"DK", 0}, {"CS", 0}, {"WZ", 0},{"HF", 0},{"GF", 0}};

    // Update InceptionDLL_v0.3.0_N+4 同类缺陷合并
    if (Save_Defects_with_Combiner) {
        std::vector<DetectionResult> merged_results = merge_detection_results_by_class(results);
        results = merged_results;
    }

    // Update InceptionDLL_v0.3.0_N+4 同类缺陷合并 Delete旧代码
    // 框体单独标注
    if (!Save_Single_Defects) {
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
    }
    else {
        for (auto& res : results) {
            draw_box(img, res, show_score, show_class, class_counter, save_or_not, image_path);
        }
    }

    return detection_results_to_string(results);
}

//Add InspectionGD v0.3.0
namespace InspectionGD {
    // ===== 算法函数实现 =====
    cv::Mat GD_Algorithms::GD_LimitTrackMaskRegion(
        const cv::Mat& mask,
        double w_area,
        double w_center,
        double w_rect) {
        cv::Mat labels, stats, centroids;
        // labels：用于存储每个像素的连通组件标签。
        // stats：用于存储每个连通区域的统计信息。是一个 N x 5 的矩阵（N 为连通区域数量），每行对应一个区域的信息，格式为 [x, y, width, height, area]：
        // x, y：区域外接矩形的左上角坐标；
        // width, height：外接矩形的宽和高；
        // area：区域包含的像素总数（面积）。
        // 用于存储每个连通区域的质心（中心点）坐标。
        int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
        if (num_labels <= 1) {
            return mask.clone();
        }

        int h = mask.rows;
        int w = mask.cols;
        int center_x = w / 2;
        int center_y = h / 2;

        std::vector<GD_RegionInfo> regions_info;
        // 收集区域特征（面积、区域中心点距离）
        for (int i = 1; i < num_labels; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            int y = stats.at<int>(i, cv::CC_STAT_TOP);
            int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

            double cx = centroids.at<double>(i, 0);
            double cy = centroids.at<double>(i, 1);

            int rect_area = width * height;
            double rectangularity = (rect_area > 0) ? (double)area / rect_area : 0.0;

            double distance = std::sqrt(std::pow(cx - center_x, 2) + std::pow(cy - center_y, 2));

            GD_RegionInfo info;
            info.label = i;
            info.area = area;
            info.rectangularity = rectangularity;
            info.distance = distance;

            regions_info.push_back(info);
        }

        // 计算归一化因子
        int max_area = 0;
        double max_rect = 0.0;
        double max_distance = 0.0;
        // 遍历获得基准值（最大面积、最接近矩形率、距中心点最近距离）
        for (const auto& region : regions_info) {
            if (region.area > max_area) max_area = region.area;
            if (region.rectangularity > max_rect) max_rect = region.rectangularity;
            if (region.distance > max_distance) max_distance = region.distance;
        }
        // 防止除零
        if (max_area == 0) max_area = 1;
        if (max_rect == 0.0) max_rect = 1.0;
        if (max_distance == 0.0) max_distance = 1.0;

        // 计算综合评价得分（按各项加权）
        for (auto& region : regions_info) {
            double area_norm = (double)region.area / max_area;
            double center_norm = 1.0 - (region.distance / max_distance);
            double rect_norm = region.rectangularity / max_rect;

            region.score = area_norm * w_area + center_norm * w_center + rect_norm * w_rect;

            region.score_detail["area_norm"] = area_norm;
            region.score_detail["center_norm"] = center_norm;
            region.score_detail["rect_norm"] = rect_norm;
        }

        // 得到最优区域
        std::sort(regions_info.begin(), regions_info.end(),
            [](const GD_RegionInfo& a, const GD_RegionInfo& b) {
                return a.score > b.score;
            });

        const auto& best = regions_info[0];

        // 创建最优区域mask
        cv::Mat new_mask = cv::Mat::zeros(mask.size(), mask.type());
        new_mask.setTo(255, labels == best.label);

        return new_mask;
    }

    std::vector<int> GD_Algorithms::GD_CalculateRegionWidths(
        const cv::Mat& img,
        const cv::Mat& mask,
        int num_regions) {

        int h = img.rows;
        int w = img.cols;
        int region_height = h / num_regions;
        std::vector<int> widths;

        for (int i = 0; i < num_regions; ++i) {
            int y1 = i * region_height;
            int y2 = (i < num_regions - 1) ? (i + 1) * region_height : h;
            int region_height_now = y2 - y1;

            cv::Mat region_rock = mask(cv::Range(y1, y2), cv::Range::all());
            int white_count = cv::countNonZero(region_rock);

            int width = (region_height_now > 0) ? std::round((double)white_count / region_height_now) : 0;
            widths.push_back(width);
        }

        return widths;
    }
    GD_AnalysisResult GD_Algorithms::GD_AnalyzeRegionWidths(
        const std::vector<int>& widths,
        double gdgk_threshold,
        double extreme_threshold,
        double cov_threshold,
        double gradient_threshold,
        double mad_threshold) {

        GD_AnalysisResult result;

        if (widths.size() < 6) {
            throw std::invalid_argument("widths列表需要至少包含6个元素才能进行有效分析");
        }

        // 1.光带过宽检测，先简单用超过25%的区域宽度超过光带过宽阈值判定，后续优化。。。11.17TODO
        int exceed_count = 0;
        for (int w : widths) {
            if (w > gdgk_threshold) exceed_count++;
        }
        double exceed_ratio = (double)exceed_count / widths.size();
        bool gdgk_result = (exceed_ratio >= 0.25);
        // 2.光带不均检测
        /*基于光带区域宽度统计学判断，加权投票决定是否异常:极值差异判断（2分）、CV变异系数阈值判断（1分）、基于Tukey's栅栏法（四分位距 IQR）的梯度差异异常值波动判断（1分）、相对中值的平均绝对偏差判断（1分），超过3分则认为光带不均；先看效果，还需要后续优化*/

        // 2.1 极值差异判断
        int diff_len = std::max(1, (int)(widths.size() * 0.1));
        std::vector<int> sorted_widths = widths;
        // 对容器 sorted_widths 中的元素进行从小到大（升序） 排序。
        std::sort(sorted_widths.begin(), sorted_widths.end());

        double avg_min = 0.0, avg_max = 0.0;
        for (int i = 0; i < diff_len; ++i) {
            avg_min += sorted_widths[i];
            avg_max += sorted_widths[widths.size() - 1 - i];
        }
        avg_min /= diff_len;
        avg_max /= diff_len;
        double extreme_diff = avg_max - avg_min;

        bool extreme_result = (extreme_diff > extreme_threshold);

        // 2.2 CV变异系数阈值判断
        double mean_width = std::accumulate(widths.begin(), widths.end(), 0.0) / widths.size();
        double variance = 0.0;
        for (int w : widths) {
            variance += std::pow(w - mean_width, 2);
        }
        double std_dev = std::sqrt(variance / widths.size());
        double cov = (mean_width > 0) ? std_dev / mean_width : 0.0;
        bool cov_result = (cov >= cov_threshold);

        //2.3 基于Tukey's方法四分位距 IQR的梯度差异异常值占比判断
        //double mean_width = std::accumulate(widths.begin(), widths.end(), 0.0) / widths.size();
        //double variance = 0.0;
        //for (int w : widths) {
        //    variance += std::pow(w - mean_width, 2);
        //}
        //double std_dev = std::sqrt(variance / widths.size());
        //double cov = (mean_width > 0) ? std_dev / mean_width : 0.0;
        //bool cov_result = (cov >= cov_threshold);

        //2.4 基于Tukey's栅栏法（四分位距 IQR）的梯度差异异常值波动判断,另外11.17决定将 Tukey；
        std::vector<int> gradients;
        for (size_t i = 1; i < widths.size(); ++i) { gradients.push_back(std::abs(widths[i] - widths[i - 1])); }

        double gradient_ratio = 0.0;
        if (!gradients.empty()) {
            std::vector<int> sorted_gradients = gradients;
            std::sort(sorted_gradients.begin(), sorted_gradients.end());

            int q1_index = gradients.size() * 0.25;
            int q3_index = gradients.size() * 0.75;
            double q1 = sorted_gradients[q1_index];
            double q3 = sorted_gradients[q3_index];
            double iqr = q3 - q1;
            //  “Tukey 中度异常值判据” 从1.5改为1，从中度改到轻度；
            double high_threshold = q3 + 1 * iqr;
            double low_threshold = q1 - 1 * iqr;

            int large_count = 0, low_count = 0;
            for (int g : gradients) {
                if (g > high_threshold) large_count++;
                if (g < low_threshold) low_count++;
            }

            gradient_ratio = (double)(large_count + low_count) / gradients.size();
        }

        bool gradient_result = (gradient_ratio > gradient_threshold);

        //2.5 相对中值的平均绝对偏差判断
        std::vector<int> sorted_for_median = widths;
        std::sort(sorted_for_median.begin(), sorted_for_median.end());
        double median_width = (widths.size() % 2 == 0) ?
            (sorted_for_median[widths.size() / 2 - 1] + sorted_for_median[widths.size() / 2]) / 2.0 :
            sorted_for_median[widths.size() / 2];

        double mad = 0.0;
        for (int w : widths) {
            mad += std::abs(w - median_width);
        }
        mad /= widths.size();

        bool mad_result = (mad > mad_threshold);

        //2 机制投票的缺陷判定
        int criteria_met = 0;
        if (extreme_result) criteria_met += 2;
        if (cov_result) criteria_met += 1;
        if (gradient_result) criteria_met += 1;
        if (mad_result) criteria_met += 1;

        bool gdbj_result = (criteria_met >= 3);

        // 保存详细结果
        result.details["extreme_diff"] = {
            {"value", extreme_diff},
            {"threshold", extreme_threshold},
            {"met", extreme_result ? 1.0 : 0.0},
            {"avg_min", avg_min},
            {"avg_max", avg_max}
        };

        result.details["cov"] = {
            {"value", cov},
            {"threshold", cov_threshold},
            {"met", cov_result ? 1.0 : 0.0},
            {"mean_width", mean_width}
        };

        result.details["gradient_ratio"] = {
            {"value", gradient_ratio},
            {"threshold", gradient_threshold},
            {"met", gradient_result ? 1.0 : 0.0}
        };

        result.details["mad"] = {
            {"value", mad},
            {"threshold", mad_threshold},
            {"met", mad_result ? 1.0 : 0.0},
            {"median_width", median_width}
        };

        result.summary["criteria_met"] = std::to_string(criteria_met);
        result.summary["total_criteria"] = "4";
        result.summary["gdgk_result"] = gdgk_result ? "true" : "false";
        result.summary["gdbj_result"] = gdbj_result ? "true" : "false";

        return result;

    }
    std::string GD_Algorithms::GD_AnaysisResult(bool gdgk_result, bool gdbj_result) {
        if (gdgk_result && gdbj_result) {
            return "GDBJ-GK";
        }
        else if (gdgk_result) {
            return "GDGK";
        }
        else if (gdbj_result) {
            return "GDBJ";
        }
        else {
            return "GDZC";
        }
    }
    void GD_Algorithms::GD_PrintAnalysisResults(
        const std::string& filename,
        const std::vector<int>& widths,
        bool gdgk_result,
        bool gdbj_result,
        const GD_AnalysisResult& analysis) {

        std::cout << "\n=== 图像分析结果: " << filename << " ===" << std::endl;

        int min_width = *std::min_element(widths.begin(), widths.end());
        int max_width = *std::max_element(widths.begin(), widths.end());
        double mean_width = std::accumulate(widths.begin(), widths.end(), 0.0) / widths.size();

        std::vector<int> sorted_widths = widths;
        std::sort(sorted_widths.begin(), sorted_widths.end());
        double median_width = (widths.size() % 2 == 0) ?
            (sorted_widths[widths.size() / 2 - 1] + sorted_widths[widths.size() / 2]) / 2.0 :
            sorted_widths[widths.size() / 2];

        std::cout << "宽度数据统计: 数量=" << widths.size()
            << ", 范围=[" << min_width << "-" << max_width << "]"
            << ", 均值=" << mean_width << ", 中位数=" << median_width << std::endl;

        // 光带过宽检测结果
        int exceed_count = 0;
        for (int w : widths) {
            if (w > 400) exceed_count++;
        }
        double exceed_ratio = (double)exceed_count / widths.size() * 100;
        std::cout << "\n[光带过宽检测]" << std::endl;
        std::cout << "超限比例: " << exceed_ratio << "% - "
            << (gdgk_result ? "⚠️ 检测到光带过宽" : "✅ 未检测到光带过宽") << std::endl;

        // 梯度异常检测详细结果
        std::cout << "\n[梯度异常检测]" << std::endl;
        auto& extreme_diff = analysis.details.at("extreme_diff");
        std::cout << "1. 极值差异: " << extreme_diff.at("value") << " (阈值: " << extreme_diff.at("threshold") << ") - "
            << (extreme_diff.at("met") ? "wrong" : "safe") << std::endl;
        std::cout << "   最小值均: " << extreme_diff.at("avg_min") << ", 最大值均: " << extreme_diff.at("avg_max") << std::endl;

        auto& cov = analysis.details.at("cov");
        std::cout << "2. 变异系数: " << cov.at("value") << " (阈值: " << cov.at("threshold") << ") - "
            << (cov.at("met") ? "wrong" : "safe") << std::endl;
        std::cout << "   平均值: " << cov.at("mean_width") << std::endl;

        auto& gradient_ratio = analysis.details.at("gradient_ratio");
        std::cout << "3. 梯度变化: " << gradient_ratio.at("value") << " (阈值: " << gradient_ratio.at("threshold") << ") - "
            << (gradient_ratio.at("met") ? "wrong" : "safe") << std::endl;

        auto& mad = analysis.details.at("mad");
        std::cout << "4. 绝对偏差: " << mad.at("value") << " (阈值: " << mad.at("threshold") << ") - "
            << (mad.at("met") ? "wrong" : "safe") << std::endl;
        std::cout << "   中位数基准: " << mad.at("median_width") << std::endl;

        // 最终结果汇总
        std::cout << "\n[检测结果汇总]" << std::endl;
        std::cout << "满足条件: " << analysis.summary.at("criteria_met") << "/" << analysis.summary.at("total_criteria") << std::endl;
        std::cout << "光带过宽: " << (gdgk_result ? "⚠️ 检测异常" : "✅ 正常") << std::endl;
        std::cout << "梯度异常: " << (gdbj_result ? "⚠️ 检测异常" : "✅ 正常") << std::endl;
        std::cout << "总体状态: " << ((gdgk_result || gdbj_result) ? "🚨 存在异常" : "✅ 正常") << std::endl;


    }

    // ====== GD_AnomalyDetector 实现======
    // 共有方法
    GD_AnomalyDetector::GD_AnomalyDetector(const InspectionGD_Config& config)
        : InspectionGD_Config_(config) {
        mroph_kernel_ = cv::Mat::ones(config.morph_kernel_size, config.morph_kernel_size, CV_8U);
    }



    GD_ProcessResult GD_AnomalyDetector::GD_AnomalyImage(cv::Mat image) {
        GD_ProcessResult result;

        // 确保图像是灰度图
        if (image.channels() != 1) {
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        }

        cv::Mat mask = createMask(image);
        cv::Mat limited_mask = limitTrackMask(mask);
        std::vector<int> widths = anomalyImageExtractWidthsOfRegion(image, limited_mask);

        GD_AnalysisResult analysis = GD_Algorithms::GD_AnalyzeRegionWidths(
            widths,
            InspectionGD_Config_.gdgk_threshold,
            InspectionGD_Config_.extreme_threshold,
            InspectionGD_Config_.cov_threshold,
            InspectionGD_Config_.gradient_threshold,
            InspectionGD_Config_.mad_threshold);

        result.gdgk_result = (analysis.summary["gdgk_result"] == "true");
        result.gdbj_result = (analysis.summary["gdbj_result"] == "true");
        result.widths = widths;
        result.analysis = analysis;
        result.category = GD_Algorithms::GD_AnaysisResult(result.gdgk_result, result.gdbj_result);

        //GD_Algorithms::GD_PrintAnalysisResults(result.filename, widths, result.gdgk_result, result.gdbj_result, analysis);

        return result;
    }

    GD_ProcessResult GD_AnomalyDetector::GD_AnomalyImage(const std::string& image_path)
    {
        cv::Mat img = preprocessImage(image_path);
        return GD_AnomalyDetector::GD_AnomalyImage(img);
    }

    std::vector<std::string> GD_AnomalyDetector::GD_AnomalyImage(const std::string& image_path, bool return_GDAnomaly_details) {
        std::vector<std::string> gd_result_info;

        try {
            // 处理图片获取详细结果
            GD_ProcessResult gd_result = GD_AnomalyImage(image_path);

            // 获取分类结果
            std::string classification_result = GD_Algorithms::GD_AnaysisResult(gd_result.gdgk_result, gd_result.gdbj_result);
            gd_result_info.push_back(classification_result);

            // 根据参数决定是否添加详细结果
            if (return_GDAnomaly_details) {
                std::string detailed_result = GD_AnomalyDetector::GD_AnomalyImage_result_with_details(gd_result);
                gd_result_info.push_back(detailed_result);
            }

            return gd_result_info;
        }
        catch (const std::exception& e) {
            std::cerr << "GD_AnomalyImage 处理异常: " << e.what() << std::endl;

            // 异常情况下返回错误信息
            gd_result_info.push_back("GDZC"); // 分类结果默认为正常

            if (return_GDAnomaly_details) {
                gd_result_info.push_back("错误: " + std::string(e.what()));
            }

            return gd_result_info;
        }
        catch (...) {
            std::cerr << "GD_AnomalyImage 处理未知异常" << std::endl;

            gd_result_info.push_back("GDZC"); // 分类结果默认为正常

            if (return_GDAnomaly_details) {
                gd_result_info.push_back("错误: 未知异常");
            }

            return gd_result_info;
        }
    }

    std::string GD_AnomalyDetector::GD_AnomalyImage_result(GD_ProcessResult& gd_result) {
        try {
            // 根据检测结果返回对应的分类字符串
            if (gd_result.gdbj_result && gd_result.gdgk_result) {
                // 双重异常情况，根据您的分类规则返回 GDBJ-GK
                return "GDBJ-GK";
            }
            else if (gd_result.gdbj_result) {
                return "GDBJ";
            }
            else if (gd_result.gdgk_result) {
                return "GDGK";
            }
            else {
                return "ZC";
            }
        }
        catch (const std::exception& e) {
            // 异常情况下返回 ZC
            std::cerr << "GD_AnomalyImage 处理异常: " << e.what() << std::endl;
            return "ZC";
        }
        catch (...) {
            // 捕获所有其他异常
            std::cerr << "GD_AnomalyImage 处理未知异常" << std::endl;
            return "ZC";
        }
    }

    std::string GD_AnomalyDetector::GD_AnomalyImage_result_with_details_CN(GD_ProcessResult& gd_result) {

        std::stringstream ss;

        // 基本信息
        ss << "=== 钢轨异常检测结果 ===" << std::endl;
        ss << "文件名: " << gd_result.filename << std::endl;
        ss << "分类结果: " << gd_result.category << std::endl;
        ss << "光带过宽检测: " << (gd_result.gdgk_result ? "异常" : "正常") << std::endl;
        ss << "梯度异常检测: " << (gd_result.gdbj_result ? "异常" : "正常") << std::endl;

        // 宽度数据统计
        if (!gd_result.widths.empty()) {
            int min_width = *std::min_element(gd_result.widths.begin(), gd_result.widths.end());
            int max_width = *std::max_element(gd_result.widths.begin(), gd_result.widths.end());
            double mean_width = std::accumulate(gd_result.widths.begin(), gd_result.widths.end(), 0.0) / gd_result.widths.size();

            std::vector<int> sorted_widths = gd_result.widths;
            std::sort(sorted_widths.begin(), sorted_widths.end());
            double median_width = (gd_result.widths.size() % 2 == 0) ?
                (sorted_widths[gd_result.widths.size() / 2 - 1] + sorted_widths[gd_result.widths.size() / 2]) / 2.0 :
                sorted_widths[gd_result.widths.size() / 2];

            ss << std::endl << "宽度数据统计:" << std::endl;
            ss << "  数量: " << gd_result.widths.size() << std::endl;
            ss << "  范围: [" << min_width << "-" << max_width << "]" << std::endl;
            ss << "  均值: " << mean_width << std::endl;
            ss << "  中位数: " << median_width << std::endl;

            // 光带过宽统计
            int exceed_count = 0;
            for (int w : gd_result.widths) {
                if (w > 400) exceed_count++; // 使用默认阈值400
            }
            double exceed_ratio = (double)exceed_count / gd_result.widths.size() * 100;
            ss << "  超限比例: " << exceed_ratio << "% (" << exceed_count << "/" << gd_result.widths.size() << ")" << std::endl;
        }

        // 详细分析结果
        if (!gd_result.analysis.details.empty()) {
            ss << std::endl << "详细分析结果:" << std::endl;

            // 极值差异
            if (gd_result.analysis.details.count("extreme_diff")) {
                auto& extreme = gd_result.analysis.details.at("extreme_diff");
                ss << "1. 极值差异: " << extreme.at("value") << " (阈值: " << extreme.at("threshold") << ") - "
                    << (extreme.at("met") ? "异常" : "正常") << std::endl;
                ss << "   最小值均: " << extreme.at("avg_min") << ", 最大值均: " << extreme.at("avg_max") << std::endl;
            }

            // 变异系数
            if (gd_result.analysis.details.count("cov")) {
                auto& cov = gd_result.analysis.details.at("cov");
                ss << "2. 变异系数: " << cov.at("value") << " (阈值: " << cov.at("threshold") << ") - "
                    << (cov.at("met") ? "异常" : "正常") << std::endl;
                ss << "   平均值: " << cov.at("mean_width") << std::endl;
            }

            // 梯度变化
            if (gd_result.analysis.details.count("gradient_ratio")) {
                auto& gradient = gd_result.analysis.details.at("gradient_ratio");
                ss << "3. 梯度变化: " << gradient.at("value") << " (阈值: " << gradient.at("threshold") << ") - "
                    << (gradient.at("met") ? "异常" : "正常") << std::endl;
            }

            // 绝对偏差
            if (gd_result.analysis.details.count("mad")) {
                auto& mad = gd_result.analysis.details.at("mad");
                ss << "4. 绝对偏差: " << mad.at("value") << " (阈值: " << mad.at("threshold") << ") - "
                    << (mad.at("met") ? "异常" : "正常") << std::endl;
                ss << "   中位数基准: " << mad.at("median_width") << std::endl;
            }
        }

        // 汇总信息
        if (!gd_result.analysis.summary.empty()) {
            ss << std::endl << "检测结果汇总:" << std::endl;
            ss << "满足条件: " << gd_result.analysis.summary.at("criteria_met")
                << "/" << gd_result.analysis.summary.at("total_criteria") << std::endl;

            // 状态图标
            std::string gdgk_status = gd_result.gdgk_result ? "⚠️ 检测异常" : "✅ 正常";
            std::string gdbj_status = gd_result.gdbj_result ? "⚠️ 检测异常" : "✅ 正常";
            std::string overall_status = (gd_result.gdgk_result || gd_result.gdbj_result) ? "🚨 存在异常" : "✅ 正常";

            ss << "光带过宽: " << gdgk_status << std::endl;
            ss << "梯度异常: " << gdbj_status << std::endl;
            ss << "总体状态: " << overall_status << std::endl;
        }

        // 原始宽度数据（可选，用于调试）
        ss << std::endl << "原始宽度数据: [";
        for (size_t i = 0; i < gd_result.widths.size(); ++i) {
            ss << gd_result.widths[i];
            if (i < gd_result.widths.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]" << std::endl;

        return ss.str();
    }
    std::string GD_AnomalyDetector::GD_AnomalyImage_result_with_details(GD_ProcessResult& gd_result) {
        std::stringstream ss;

        // Basic Information
        ss << "=== Rail Anomaly Detection Result ===" << std::endl;
        ss << "Filename: " << gd_result.filename << std::endl;
        ss << "Classification: " << gd_result.category << std::endl;
        ss << "GDGK Detection: " << (gd_result.gdgk_result ? "Anomaly" : "Normal") << std::endl;
        ss << "GDBJ Detection: " << (gd_result.gdbj_result ? "Anomaly" : "Normal") << std::endl;

        // Width Data Statistics
        if (!gd_result.widths.empty()) {
            int min_width = *std::min_element(gd_result.widths.begin(), gd_result.widths.end());
            int max_width = *std::max_element(gd_result.widths.begin(), gd_result.widths.end());
            double mean_width = std::accumulate(gd_result.widths.begin(), gd_result.widths.end(), 0.0) / gd_result.widths.size();

            std::vector<int> sorted_widths = gd_result.widths;
            std::sort(sorted_widths.begin(), sorted_widths.end());
            double median_width = (gd_result.widths.size() % 2 == 0) ?
                (sorted_widths[gd_result.widths.size() / 2 - 1] + sorted_widths[gd_result.widths.size() / 2]) / 2.0 :
                sorted_widths[gd_result.widths.size() / 2];

            ss << std::endl << "Width Data Statistics:" << std::endl;
            ss << "  Count: " << gd_result.widths.size() << std::endl;
            ss << "  Range: [" << min_width << "-" << max_width << "]" << std::endl;
            ss << "  Mean: " << mean_width << std::endl;
            ss << "  Median: " << median_width << std::endl;

            // GDGK Exceed Statistics
            int exceed_count = 0;
            for (int w : gd_result.widths) {
                if (w > 400) exceed_count++; // Using default threshold 400
            }
            double exceed_ratio = (double)exceed_count / gd_result.widths.size() * 100;
            ss << "  Exceed Ratio: " << exceed_ratio << "% (" << exceed_count << "/" << gd_result.widths.size() << ")" << std::endl;
        }

        // Detailed Analysis Results
        if (!gd_result.analysis.details.empty()) {
            ss << std::endl << "Detailed Analysis Results:" << std::endl;

            // Extreme Difference
            if (gd_result.analysis.details.count("extreme_diff")) {
                auto& extreme = gd_result.analysis.details.at("extreme_diff");
                ss << "1. Extreme Difference: " << extreme.at("value") << " (Threshold: " << extreme.at("threshold") << ") - "
                    << (extreme.at("met") ? "Anomaly" : "Normal") << std::endl;
                ss << "   Min Average: " << extreme.at("avg_min") << ", Max Average: " << extreme.at("avg_max") << std::endl;
            }

            // Coefficient of Variation
            if (gd_result.analysis.details.count("cov")) {
                auto& cov = gd_result.analysis.details.at("cov");
                ss << "2. Coefficient of Variation: " << cov.at("value") << " (Threshold: " << cov.at("threshold") << ") - "
                    << (cov.at("met") ? "Anomaly" : "Normal") << std::endl;
                ss << "   Mean Width: " << cov.at("mean_width") << std::endl;
            }

            // Gradient Ratio
            if (gd_result.analysis.details.count("gradient_ratio")) {
                auto& gradient = gd_result.analysis.details.at("gradient_ratio");
                ss << "3. Gradient Ratio: " << gradient.at("value") << " (Threshold: " << gradient.at("threshold") << ") - "
                    << (gradient.at("met") ? "Anomaly" : "Normal") << std::endl;
            }

            // Mean Absolute Deviation
            if (gd_result.analysis.details.count("mad")) {
                auto& mad = gd_result.analysis.details.at("mad");
                ss << "4. Mean Absolute Deviation: " << mad.at("value") << " (Threshold: " << mad.at("threshold") << ") - "
                    << (mad.at("met") ? "Anomaly" : "Normal") << std::endl;
                ss << "   Median Baseline: " << mad.at("median_width") << std::endl;
            }
        }

        // Summary Information
        if (!gd_result.analysis.summary.empty()) {
            ss << std::endl << "Detection Result Summary:" << std::endl;
            ss << "Criteria Met: " << gd_result.analysis.summary.at("criteria_met")
                << "/" << gd_result.analysis.summary.at("total_criteria") << std::endl;

            // Status indicators
            std::string gdgk_status = gd_result.gdgk_result ? " Anomaly Detected" : "Normal";
            std::string gdbj_status = gd_result.gdbj_result ? " Anomaly Detected" : "Normal";
            std::string overall_status = (gd_result.gdgk_result || gd_result.gdbj_result) ? "Anomaly Exists" : " Normal";

            ss << "GDGK: " << gdgk_status << std::endl;
            ss << "GDBJ: " << gdbj_status << std::endl;
            ss << "Overall Status: " << overall_status << std::endl;
        }

        // Original Width Data (optional, for debugging)
        ss << std::endl << "Original Width Data: [";
        for (size_t i = 0; i < gd_result.widths.size(); ++i) {
            ss << gd_result.widths[i];
            if (i < gd_result.widths.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]" << std::endl;

        return ss.str();
    }
    std::string GD_AnomalyDetector::SaveGD_Image(
        cv::Mat image,
        bool gdgk_result,
        bool gdbj_result,
        const std::string& source_path,
        const std::string& target_folder) {

        std::string category = GD_Algorithms::GD_AnaysisResult(gdgk_result, gdbj_result);
        std::string target_path = target_folder + "/" + category;
        // 创建目录（如果不存在）
        if (!fs::exists(target_path)) {
            // 尝试创建目录（包括所有父目录）
            bool created = fs::create_directories(target_path);
            if (!created) {
                // 创建失败时抛出异常或处理错误
                throw std::runtime_error("Failed to create directory: " + target_path);
                // 或输出错误信息：
                // std::cerr << "Error: Failed to create directory " << target_path << std::endl;
            }
        }

        std::string filename = source_path.substr(source_path.find_last_of("/\\") + 1);
        std::string full_path = target_path + "/" + filename;

        // 保存图像
        cv::imwrite(full_path, image);
        std::cout << "图片 " << filename << " 已分类到: " << category << std::endl;

        return category;
    }

    // 私有方法
    cv::Mat GD_AnomalyDetector::preprocessImage(const std::string& image_path) {
        cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            throw std::runtime_error("无法加载图片: " + image_path);
        }
        return img;
    }
    cv::Mat GD_AnomalyDetector::createMask(const cv::Mat& gray_image) {
        cv::Mat mask;
        cv::threshold(gray_image, mask, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        return mask;
    }
    cv::Mat GD_AnomalyDetector::limitTrackMask(const cv::Mat& mask) {
        cv::Mat mask_closed;
        cv::morphologyEx(mask, mask_closed, cv::MORPH_CLOSE, mroph_kernel_);

        cv::Mat mask_limited = GD_Algorithms::GD_LimitTrackMaskRegion(
            mask_closed,
            InspectionGD_Config_.w_area,
            InspectionGD_Config_.w_center,
            InspectionGD_Config_.w_rect);

        return mask_limited;
    }
    std::vector<int> GD_AnomalyDetector::anomalyImageExtractWidthsOfRegion(const cv::Mat& img, const cv::Mat& mask) {
        return GD_Algorithms::GD_CalculateRegionWidths(img, mask, InspectionGD_Config_.num_regions);
    }
}

//Add Trackinspection2D_System_v0.3.0_N+5 添加namespace DefectResultCompletionUtils
namespace DefectResultCompletionUtils {
    // 辅助函数：解析文件名，提取原始图像名、部分索引和拉伸倍数
    bool parseImageName(const std::string& imageName, std::string& sourceName, int& partIndex, int& stretchFactor, int& offsetX) {
        // 格式示例: "00018_4of2_DK1.jpeg"
        // 原始图像名: 00018.jpeg
        // 4of2 表示拉伸4倍后切分的4部分中的第2部分

        size_t extPos = imageName.find_last_of('.');
        std::string nameWithoutExt = (extPos != std::string::npos) ?
            imageName.substr(0, extPos) : imageName;

        // 查找 "of" 位置
        size_t ofPos = nameWithoutExt.find("of");
        if (ofPos == std::string::npos || ofPos < 2) return false;

        // 查找拉伸倍数前面的下划线
        size_t underscorePos = nameWithoutExt.find_last_of('_', ofPos - 2);
        if (underscorePos == std::string::npos) return false;

        // 提取拉伸倍数
        std::string stretchStr = nameWithoutExt.substr(underscorePos + 1, ofPos - underscorePos - 1);
        // 提取部分索引
        std::string indexStr = nameWithoutExt.substr(ofPos + 2);

        // 移除可能的后缀（如_DK1）
        size_t nextUnderscore = indexStr.find('_');
        if (nextUnderscore != std::string::npos) {
            // 如果后缀中包含偏移信息，可以在这里解析
            // 例如: "2_DK1" -> 部分索引是2
            indexStr = indexStr.substr(0, nextUnderscore);
        }

        try {
            stretchFactor = std::stoi(stretchStr);
            partIndex = std::stoi(indexStr);
            // 原始图像名是第一个下划线前的部分
            sourceName = nameWithoutExt.substr(0, underscorePos) + ".jpeg";
            return true;
        }
        catch (...) {
            return false;
        }
    }
    std::vector<std::vector<std::pair<int, int>>> parseMultiPolygons(const std::string& pointsStr) {
        std::vector<std::vector<std::pair<int, int>>> polygons;

        if (pointsStr.empty() || pointsStr == "[]" || pointsStr == "[[]]") {
            return polygons;
        }

        size_t pos = 0;

        // 跳过最外层的 [
        while (pos < pointsStr.size() && pointsStr[pos] != '[') pos++;
        if (pos < pointsStr.size()) pos++;

        while (pos < pointsStr.size()) {
            // 跳过空白
            while (pos < pointsStr.size() && (pointsStr[pos] == ' ' || pointsStr[pos] == '\n')) {
                pos++;
            }

            if (pointsStr[pos] == '[') {
                // 开始一个新多边形
                std::vector<std::pair<int, int>> currentPolygon;
                pos++; // 跳过 [

                while (pos < pointsStr.size() && pointsStr[pos] != ']') {
                    // 跳过空白
                    while (pos < pointsStr.size() && (pointsStr[pos] == ' ' || pointsStr[pos] == '\n')) {
                        pos++;
                    }

                    if (pointsStr[pos] == '[') {
                        pos++; // 跳过 [

                        // 解析x
                        std::string xStr;
                        while (pos < pointsStr.size() && isdigit(pointsStr[pos])) {
                            xStr += pointsStr[pos];
                            pos++;
                        }

                        // 跳过逗号
                        if (pos < pointsStr.size() && pointsStr[pos] == ',') {
                            pos++;
                        }

                        // 解析y
                        std::string yStr;
                        while (pos < pointsStr.size() && isdigit(pointsStr[pos])) {
                            yStr += pointsStr[pos];
                            pos++;
                        }

                        // 跳过 ]
                        if (pos < pointsStr.size() && pointsStr[pos] == ']') {
                            pos++;
                        }

                        if (!xStr.empty() && !yStr.empty()) {
                            try {
                                int x = std::stoi(xStr);
                                int y = std::stoi(yStr);
                                currentPolygon.emplace_back(x, y);
                            }
                            catch (...) {
                                // 忽略转换错误
                            }
                        }

                        // 跳过点之间的逗号
                        if (pos < pointsStr.size() && pointsStr[pos] == ',') {
                            pos++;
                        }
                    }
                    else {
                        pos++;
                    }
                }

                // 保存多边形
                if (!currentPolygon.empty()) {
                    polygons.push_back(currentPolygon);
                }

                // 跳过 ]
                if (pos < pointsStr.size() && pointsStr[pos] == ']') {
                    pos++;
                }

                // 跳过多边形之间的逗号
                if (pos < pointsStr.size() && pointsStr[pos] == ',') {
                    pos++;
                }
            }
            else if (pointsStr[pos] == ']') {
                // 结束整个数组
                break;
            }
            else {
                pos++;
            }
        }

        return polygons;
    }
    // 辅助函数：解析Points字符串
    std::vector<std::pair<int, int>> parsePoints(const std::string& pointsStr) {
        // 为了兼容性，调用新的解析函数然后展平
        auto polygons = parseMultiPolygons(pointsStr);
        std::vector<std::pair<int, int>> allPoints;

        for (const auto& polygon : polygons) {
            allPoints.insert(allPoints.end(), polygon.begin(), polygon.end());
        }

        return allPoints;
    }
    // 辅助函数：序列化多个多边形为字符串
    std::string serializeMultiPolygons(const std::vector<std::vector<std::pair<int, int>>>& polygons) {
        if (polygons.empty()) return "[]";

        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < polygons.size(); ++i) {
            const auto& polygon = polygons[i];
            ss << "[";
            for (size_t j = 0; j < polygon.size(); ++j) {
                ss << "[" << polygon[j].first << "," << polygon[j].second << "]";
                if (j < polygon.size() - 1) ss << ",";
            }
            ss << "]";
            if (i < polygons.size() - 1) ss << ",";
        }
        ss << "]";
        return ss.str();
    }
    // 辅助函数：序列化Points为字符串
    std::string serializePoints(const std::vector<std::pair<int, int>>& points) {
        // 为了兼容性，将单层数组包装成单个多边形
        if (points.empty()) return "[]";

        std::stringstream ss;
        ss << "[[";
        for (size_t i = 0; i < points.size(); ++i) {
            ss << "[" << points[i].first << "," << points[i].second << "]";
            if (i < points.size() - 1) ss << ",";
        }
        ss << "]]";
        return ss.str();
    }
    INCEPTIONDLL_API void completeSourceInfo(DefectResult& dr, int originalHeight,int originalWidth) {
        // 如果已经补全过，直接返回
        if (!dr.Source_ImageName.empty() && dr.Source_X != -1 && dr.Source_Y != -1) {
            return;
        }
        std::string sourceImageName;
        int partIndex = 0;
        int stretchFactor = 0;
        // 解析图像名
        if (!parseImageName(dr.ImageName, sourceImageName, partIndex, stretchFactor,dr.offset_x)) {
            // 解析失败，使用默认值
            dr.Source_ImageName = "unknown";
            return;
        }
        // 设置原始图像名
        dr.Source_ImageName = sourceImageName;
        // 检查是否存在缺陷坐标信息
        bool hasDefectCoords = (dr.X != -1 && dr.Y != -1 && dr.W != -1 && dr.H != -1);
        if (!hasDefectCoords) {
            //没有缺陷坐标，使用拉伸裁切图在原图中的坐标
            //计算拉伸裁切图在原图中的位置
            float sourceX = 0;  // 假设裁切从X=0开始
            float sourceY = 0;
            float sourceW = (originalWidth > 0) ? originalWidth : 0;
            float sourceH = (originalHeight > 0) ? (originalHeight / stretchFactor) : 0;
            // 计算Y坐标：第partIndex部分在拉伸图中的起始位置
            if (originalHeight > 0) {
                // 每部分在原始图像中的高度
                float partHeightInOriginal = originalHeight / stretchFactor;
                sourceY = (partIndex - 1) * partHeightInOriginal;

                // 如果有offset_x，需要考虑裁切
                if (dr.offset_x != -1) {
                    sourceX = static_cast<float>(dr.offset_x);
                }

            }
            dr.Source_X = static_cast<int>(std::round(sourceX));
            dr.Source_Y = static_cast<int>(std::round(sourceY));
            dr.Source_W = static_cast<int>(std::round(sourceW));
            dr.Source_H = static_cast<int>(std::round(sourceH));
            // 没有缺陷点，所以Source_Points为空
            dr.Source_Points = "[]";
        }
        else {
            //存在缺陷坐标，计算缺陷在原图中的坐标
            // 步骤1：当前缺陷在裁切图中的坐标 (dr.X, dr.Y, dr.W, dr.H)
            // 步骤2：裁切图来自第partIndex部分图，裁切偏移为offset_x
            // 步骤3：第partIndex部分图来自原图纵向拉伸stretchFactor倍后切分的stretchFactor部分之一
            float partX = dr.X + (dr.offset_x != -1 ? static_cast<float>(dr.offset_x) : 0);  // X坐标不变
            float partY = dr.Y;
            float partHeightInStretch = (originalHeight > 0) ? (originalHeight * stretchFactor / stretchFactor) : 0;
            float stretchY = partY + (partIndex - 1) * partHeightInStretch;
            // 最后计算在原图中的坐标（反向拉伸）
            dr.Source_Y = static_cast<int>(std::round(stretchY / stretchFactor));
            dr.Source_X = static_cast<int>(std::round(partX));
            // 计算尺寸（注意：宽度不变，高度需要反向拉伸）
            dr.Source_H = static_cast<int>(std::round(dr.H / stretchFactor));
            dr.Source_W = static_cast<int>(std::round(dr.W));
            if (!dr.Points.empty() && dr.Points != "[]" && dr.Points != "[[]]") {
                // 使用parseMultiPolygons函数解析Points字符串
                auto polygons = parseMultiPolygons(dr.Points);
                std::vector<std::vector<std::pair<int, int>>> sourcePolygons;
                for (const auto& polygon : polygons) {
                    std::vector<std::pair<int, int>> sourcePolygon;

                    for (const auto& point : polygon) {
                        // Points是相对于缺陷框内部的坐标
                        // 首先转换为裁切图中的绝对坐标
                        float absX = dr.X + point.first;
                        float absY = dr.Y + point.second;
                        // 转换为在第partIndex部分图中的绝对坐标
                        float absPartX = absX + (dr.offset_x != -1 ? static_cast<float>(dr.offset_x) : 0);
                        float absPartY = absY;
                        // 转换为在拉伸图中的绝对坐标
                        // 第partIndex部分图在拉伸图中的起始Y坐标
                        float stretchStartY = (partIndex - 1) * originalHeight;
                        float absStretchY = absPartY + stretchStartY;
                        // 转换为原始图像中的绝对坐标（反向拉伸）
                        float sourceAbsY = absStretchY / stretchFactor;
                        float sourceAbsX = absPartX;  // X坐标不需要反向拉伸
                        // 转换为相对于原始缺陷框的坐标
                        float sourceRelX = sourceAbsX - dr.Source_X;
                        float sourceRelY = sourceAbsY - dr.Source_Y;
                        // 四舍五入取整
                        sourcePolygon.emplace_back(
                            static_cast<int>(std::round(sourceRelX)),
                            static_cast<int>(std::round(sourceRelY))
                        );
                    }
                    sourcePolygons.push_back(sourcePolygon);
                }
                // 使用serializeMultiPolygons函数将点序列化为字符串
                dr.Source_Points = serializeMultiPolygons(sourcePolygons);
            }
            else {
                dr.Source_Points = "[]";
            }
        }
    }
    INCEPTIONDLL_API void completeSourceInfo(DefectResult_with_position& dr, int originalHeight, int originalWidth) {
        // 如果已经补全过，直接返回
        if (!dr.Source_ImageName.empty() && dr.Source_X != -1 && dr.Source_Y != -1) {
            return;
        }
        std::string sourceImageName;
        int partIndex = 0;
        int stretchFactor = 0;
        // 解析图像名
        if (!parseImageName(dr.ImageName, sourceImageName, partIndex, stretchFactor, dr.offset_x)) {
            // 解析失败，使用默认值
            dr.Source_ImageName = "unknown";
            return;
        }
        // 设置原始图像名
        dr.Source_ImageName = sourceImageName;
        // 检查是否存在缺陷坐标信息
        bool hasDefectCoords = (dr.X != -1 && dr.Y != -1 && dr.W != -1 && dr.H != -1);
        if (!hasDefectCoords) {
            //没有缺陷坐标，使用拉伸裁切图在原图中的坐标
            //计算拉伸裁切图在原图中的位置
            float sourceX = 0;  // 假设裁切从X=0开始
            float sourceY = 0;
            float sourceW = (originalWidth > 0) ? originalWidth : 0;
            float sourceH = (originalHeight > 0) ? (originalHeight / stretchFactor) : 0;
            // 计算Y坐标：第partIndex部分在拉伸图中的起始位置
            if (originalHeight > 0) {
                // 每部分在原始图像中的高度
                float partHeightInOriginal = originalHeight / stretchFactor;
                sourceY = (partIndex - 1) * partHeightInOriginal;

                // 如果有offset_x，需要考虑裁切
                if (dr.offset_x != -1) {
                    sourceX = static_cast<float>(dr.offset_x);
                }

            }
            dr.Source_X = static_cast<int>(std::round(sourceX));
            dr.Source_Y = static_cast<int>(std::round(sourceY));
            dr.Source_W = static_cast<int>(std::round(sourceW));
            dr.Source_H = static_cast<int>(std::round(sourceH));
            // 没有缺陷点，所以Source_Points为空
            dr.Source_Points = "[]";
        }
        else {
            //存在缺陷坐标，计算缺陷在原图中的坐标
            // 步骤1：当前缺陷在裁切图中的坐标 (dr.X, dr.Y, dr.W, dr.H)
            // 步骤2：裁切图来自第partIndex部分图，裁切偏移为offset_x
            // 步骤3：第partIndex部分图来自原图纵向拉伸stretchFactor倍后切分的stretchFactor部分之一
            float partX = dr.X + (dr.offset_x != -1 ? static_cast<float>(dr.offset_x) : 0);  // X坐标不变
            float partY = dr.Y;
            float partHeightInStretch = (originalHeight > 0) ? (originalHeight * stretchFactor / stretchFactor) : 0;
            float stretchY = partY + (partIndex - 1) * partHeightInStretch;
            // 最后计算在原图中的坐标（反向拉伸）
            dr.Source_Y = static_cast<int>(std::round(stretchY / stretchFactor));
            dr.Source_X = static_cast<int>(std::round(partX));
            // 计算尺寸（注意：宽度不变，高度需要反向拉伸）
            dr.Source_H = static_cast<int>(std::round(dr.H / stretchFactor));
            dr.Source_W = static_cast<int>(std::round(dr.W));
            if (!dr.Points.empty() && dr.Points != "[]" && dr.Points != "[[]]") {
                auto polygons = parseMultiPolygons(dr.Points);
                std::vector<std::vector<std::pair<int, int>>> sourcePolygons;

                for (const  auto& polygon : polygons) {
                    std::vector<std::pair<int, int>> sourcePolygon;

                    for (const auto& point : polygon) {
                        // Points是相对于缺陷框内部的坐标
                        // 首先转换为裁切图中的绝对坐标
                        float absX = dr.X + point.first;
                        float absY = dr.Y + point.second;
                        // 转换为在第partIndex部分图中的绝对坐标
                        float absPartX = absX + (dr.offset_x != -1 ? static_cast<float>(dr.offset_x) : 0);
                        float absPartY = absY;
                        // 转换为在拉伸图中的绝对坐标
                        // 第partIndex部分图在拉伸图中的起始Y坐标
                        float stretchStartY = (partIndex - 1) * originalHeight;
                        float absStretchY = absPartY + stretchStartY;
                        // 转换为原始图像中的绝对坐标（反向拉伸）
                        float sourceAbsY = absStretchY / stretchFactor;
                        float sourceAbsX = absPartX;  // X坐标不需要反向拉伸
                        // 转换为相对于原始缺陷框的坐标
                        float sourceRelX = sourceAbsX - dr.Source_X;
                        float sourceRelY = sourceAbsY - dr.Source_Y;
                        // 四舍五入取整
                        sourcePolygon.emplace_back(
                            static_cast<int>(std::round(sourceRelX)),
                            static_cast<int>(std::round(sourceRelY))
                        );
                    }
                    sourcePolygons.push_back(sourcePolygon);
                }
                // 使用serializePoints函数将点序列化为字符串
                dr.Source_Points = serializeMultiPolygons(sourcePolygons);
            }
            else {
                dr.Source_Points = "[]";
            }
        }
    }
    // 批量处理函数
    INCEPTIONDLL_API void completeSourceInfoForAll(std::vector<DefectResult>& results, int originalHeight, int originalWidth) {
        for (auto& dr : results) {
            completeSourceInfo(dr, originalHeight);
        }
    }
    INCEPTIONDLL_API void completeSourceInfoForAll(std::vector<DefectResult_with_position>& results, int originalHeight, int originalWidth) {
        for (auto& dr : results) {
            completeSourceInfo(dr, originalHeight);
        }
    }
}