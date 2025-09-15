// Add v0.3.0_n+6 将此前Inception_main_onnx_v0.3.0.cpp内的所有辅助函数和关键函数放入一个新的Inspction.h/cpp
#pragma once
#ifndef INSPECTION_H
#define INSPECTION_H

#include <string>
#include <vector>
#include <optional>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <onnxruntime_cxx_api.h>
#include <atomic>
#include <mutex>
#include <regex>
#include <iostream>
#include <unordered_set> 
#include <future>
#include <map>
#include <set>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "sqlite_loader.h"
#include "xml_loader.h"
#include "csv_loader.h"
#include "InceptionDLL_v0.3.0.h"
#include "InceptionUtils_v0.3.0.h"
#include <nlohmann/json.hpp>

using namespace std;
namespace fs = std::filesystem;

// 前向声明
struct SensorData;
struct DefectResult;
struct DefectResult_with_position;

// 外部变量声明
extern std::mutex cout_mutex;
#define MAX_THREADS 8
extern bool TestModel;
extern int Source_IMG_X;
extern int Source_IMG_Y;
extern int IMG_SIZE; 
#define IMG_SIZE 256
// 其他可能需要的全局变量
extern SQLiteLoader g_sqliteLoader;  

// ==================== 内连辅助功能函数 ===========================

/**
 * @brief 格式化持续时间显示
 * @param duration 持续时间（秒）
 * @return std::string 格式化后的字符串（如"5分30秒"）
 */
inline std::string format_duration(const std::chrono::seconds& duration) {
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    auto seconds = duration - minutes;
    return std::to_string(minutes.count()) + " 分 " + std::to_string(seconds.count()) + " 秒";
}

/**
 * @brief 解析图像文件名格式
 * @param imageName 图像文件名（格式如"000001_2of3.jpeg"）
 * @param imgIndex 输出参数：图像索引（如000001）
 * @param part 输出参数：当前部分编号
 * @param totalParts 输出参数：总部分数
 * @return bool 解析成功返回true，否则返回false
 */
inline bool parseImageName(const std::string& imageName, int& imgIndex, int& part, int& totalParts) {
    std::regex pattern(R"((\d+)_(\d+)of(\d+)\.jpeg)");
    std::smatch matches;
    if (std::regex_match(imageName, matches, pattern)) {
        imgIndex = std::stoi(matches[1].str());
        part = std::stoi(matches[2].str());
        totalParts = std::stoi(matches[3].str());
        return true;
    }
    return false;
}

/**
 * @brief 解析逗号分隔的类别列表字符串
 * @param input 输入字符串（格式如"GD, BM, ZC"）
 * @return std::vector<std::string> 解析后的类别列表
 * @details 自动去除空格，跳过空字符串
 */
inline std::vector<std::string> parseClassList(const std::string& input) {
    std::vector<std::string> classes;
    std::stringstream ss(input);
    std::string cls;

    // 按逗号分割字符串
    while (std::getline(ss, cls, ',')) {
        // 去除头部空格
        auto start = std::find_if(cls.begin(), cls.end(), [](int ch) {
            return !std::isspace(ch);
            });
        // 去除尾部空格
        auto end = std::find_if(cls.rbegin(), cls.rend(), [](int ch) {
            return !std::isspace(ch);
            }).base();

        // 提取有效子串并添加到结果（跳过空字符串）
        if (start < end) {
            classes.emplace_back(start, end);
        }
    }

    return classes;
}

/**
 * @brief 验证UTF-8字符串有效性
 * @param str 待验证的字符串
 * @return bool 有效返回true，否则返回false
 */
inline bool is_valid_utf8(const std::string& str) {
    int c, i, ix, n, j;
    for (i = 0, ix = str.length(); i < ix; i++) {
        c = (unsigned char)str[i];
        if (0x00 <= c && c <= 0x7F) n = 0; // 0bbbbbbb
        else if ((c & 0xE0) == 0xC0) n = 1; // 110bbbbb
        else if (c == 0xED && i + 1 < ix && ((unsigned char)str[i + 1] & 0xA0) == 0xA0)
            return false; // UTF-16 surrogate half
        else if ((c & 0xF0) == 0xE0) n = 2; // 1110bbbb
        else if ((c & 0xF8) == 0xF0) n = 3; // 11110bbb
        else return false;
        for (j = 0; j < n && i < ix; j++) {
            if ((++i == ix) || ((str[i] & 0xC0) != 0x80))
                return false;
        }
    }
    return true;
}
/**
 * @brief 宽字符串转换为UTF-8字符串
 * @param wstr 输入宽字符串
 * @return std::string UTF-8编码的字符串
 * @throws std::runtime_error 转换失败时抛出异常
 */
inline std::string wstring_to_utf8(const std::wstring& wstr) {
    if (wstr.empty()) return {};
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (size_needed == 0) {
        throw std::runtime_error("无法将 wstring 转换为 UTF-8: 无法映射的字符");
    }
    std::string str(size_needed - 1, '\0'); // 去掉结尾的 '\0'
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &str[0], size_needed, nullptr, nullptr);
    return str;
}

/**
 * @brief GBK编码转换为UTF-8编码
 * @param gbkStr GBK编码的字符串
 * @return std::string UTF-8编码的字符串
 */
inline std::string GbkToUtf8(const std::string& gbkStr) {
    int len = MultiByteToWideChar(CP_ACP, 0, gbkStr.c_str(), -1, nullptr, 0);
    std::wstring wstr(len, L'\0');
    MultiByteToWideChar(CP_ACP, 0, gbkStr.c_str(), -1, &wstr[0], len);

    len = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string utf8str(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &utf8str[0], len, nullptr, nullptr);

    // 去掉最后的\0
    if (!utf8str.empty() && utf8str.back() == '\0') utf8str.pop_back();
    return utf8str;
}

/**
 * @brief 按指定分隔符分割字符串
 * @param s 输入字符串
 * @param delimiter 分隔符
 * @return std::vector<std::string> 分割后的字符串列表
 */
inline std::vector<std::string> split_name(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}
// 从ImageName中提取基础文件名（如000001.jpeg）
/**
 * @brief 从图像文件名中提取基础文件名
 * @param imageName 完整图像文件名（如"000001_4of3_DK1.jpeg"）
 * @return std::string 基础文件名（如"000001.jpeg"）
 * @details 去除"_数字of数字"和相机标识等后缀信息
 */
inline std::string getBaseImageName(const std::string& imageName) {
    // 1. 分离文件名和扩展名（假设扩展名固定为.jpeg）
    size_t dotPos = imageName.find_last_of('.');
    if (dotPos == std::string::npos) {
        return imageName; // 异常情况：无扩展名，直接返回原名称
    }
    std::string ext = imageName.substr(dotPos); // 得到 ".jpeg"

    // 2. 提取文件名主体（不含扩展名）
    std::string baseWithoutExt = imageName.substr(0, dotPos); // 如 "000001_4of3" 或 "000003_4of1_DK1"

    // 3. 按 '_' 分割，取第一个片段作为基础名称
    std::vector<std::string> parts = split_name(baseWithoutExt, '_');
    if (parts.empty()) {
        return imageName; // 异常情况：无分割结果，返回原名称
    }
    std::string baseName = parts[0]; // 得到 "000001" 或 "000003"

    // 4. 拼接基础名称和扩展名
    return baseName + ext; // 最终结果："000001.jpeg" 或 "000003.jpeg"
}
// 数据采集保存函数（适配XML格式，解决变量访问问题）
// v0.2.9 修复数据采集保存函数的Unicode路径问题
// v0.3.0 优化了数据采集保存原图像，ADD
// 分割字符串的工具函数（按指定分隔符分割）
/**
 * @brief 收集目标类别的图像到指定目录
 * @tparam T 缺陷结果类型（DefectResult或DefectResult_with_position）
 * @param results 缺陷检测结果列表
 * @param img_root_path 原始图像根目录
 * @param target_classes 目标类别列表
 * @param data_collect_enabled 是否启用数据收集
 * @param TestModel_Flag 测试模式标志
 * @param L_cam_name 左相机显示名称
 * @param R_cam_name 右相机显示名称
 * @details 将指定类别的缺陷图像复制到Data_Collector目录，按类别分类保存
 */
template<typename T>
void collect_target_images(const std::vector<T>& results, const std::string& img_root_path,
    const std::vector<std::string>& target_classes, bool data_collect_enabled,
    bool TestModel_Flag, const std::string& L_cam_name, const std::string& R_cam_name) {
    if (!data_collect_enabled || target_classes.empty() || results.empty()) {
        return;
    }

    fs::path root_path = fs::path(img_root_path).root_path();
    std::string base_save_dir = root_path.string() + "Data_Collector";
    fs::create_directories(base_save_dir);

    for (const auto& result : results) {
        auto cls_it = std::find(target_classes.begin(), target_classes.end(), result.DefectType);
        if (cls_it == target_classes.end()) {
            continue;
        }
        std::string current_cls = *cls_it;

        std::string cam_name = (result.Camera == "L") ? "DATL_railhead_stretch" : "DATR_railhead_stretch";
        //std::cout << "转换前 cam_name: " << cam_name << std::endl;
         //修正乱码与中文的对应关系
        if (cam_name == "鍙崇浉鏈") {
            cam_name = "右相机";
        }
        else if (cam_name == "宸︾浉鏈") {
            cam_name = "左相机";
        }
        //std::cout << "转换后 cam_name: " << cam_name << std::endl;
        if (result.DefectType == "BM" && result.DefectType == "ZC" && result.DefectType == "GD") {
            std::string full_img_path = img_root_path + "\\" + cam_name + "\\" + result.ImageName;
            std::string full_source_img_path = img_root_path + "\\" + cam_name + "\\" + getBaseImageName(result.ImageName);
        }
        else {
            std::string full_img_path = img_root_path + "\\" + cam_name + "\\" + "detection_result\\" + result.ImageName;
        }
        std::string full_img_path = img_root_path + "\\" + cam_name + "\\" + result.ImageName;
        //std::cout << "img_root_path: " << img_root_path << std::endl;
        cam_name = GbkToUtf8(cam_name);
        //std::cout << "转换后 cam_name: " << cam_name << std::endl;
        //std::cout << "result.ImageName: " << result.ImageName << std::endl;

        std::wstring w_img_root_path = std::wstring(img_root_path.begin(), img_root_path.end());
        std::wstring w_cam_name = std::wstring(cam_name.begin(), cam_name.end());
        std::wstring w_image_name = std::wstring(result.ImageName.begin(), result.ImageName.end());

        std::wstring w_full_img_path = w_img_root_path + L"\\" + w_cam_name + L"\\" + w_image_name;
        std::filesystem::path full_img_wpath(w_full_img_path);

        if (!fs::exists(full_img_path)) {
            if (TestModel_Flag) {
                std::cerr << "警告：图像不存在 " << full_img_path << std::endl;
            }
            std::cout << "警告：图像不存在 " << full_img_path << std::endl;
            continue;
        }

        cv::Mat img = cv::imread(full_img_path, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            if (TestModel_Flag) {
                std::cerr << "警告：无法读取图像 " << full_img_wpath << std::endl;
            }
            continue;
        }

        std::string cls_save_dir = base_save_dir + "/" + current_cls;
        fs::create_directories(cls_save_dir);

        std::string datetime_info = fs::path(img_root_path).filename().string();
        std::string new_filename = datetime_info + "_" + result.Camera + "_" + result.ImageName;
        std::string dest_path = cls_save_dir + "/" + new_filename;

        try {
            fs::path dest_fs_path(dest_path);
            std::wstring w_dest_path = dest_fs_path.wstring();

            // 转换为 UTF-8 编码的 std::string
            std::string utf8_dest_path = wstring_to_utf8(w_dest_path);

            // 使用 UTF-8 路径保存图像
            if (!cv::imwrite(utf8_dest_path, img)) {
                if (TestModel_Flag) {
                    std::cerr << "警告：保存失败 " << utf8_dest_path << std::endl;
                }
            }
            else {
                if (TestModel_Flag) {
                    std::cout << "数据采集：类别[" << current_cls << "] 保存至 " << utf8_dest_path << std::endl;
                }
            }
            std::cout << "数据采集：类别[" << current_cls << "] 保存至 " << utf8_dest_path << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "保存图片异常: " << e.what() << std::endl;
            continue;
        }
    }
}
// ==================== 内连辅助功能函数 ========================== =
/**
 * @brief 计算缺陷的里程位置
 * @param csvData CSV传感器数据
 * @param defect 缺陷检测结果
 * @param imageHeight 图像高度（默认1024）
 * @return DefectResult_with_position 包含位置信息的缺陷结果
 * @details 根据传感器数据和图像中的Y坐标计算缺陷的实际里程位置
 */
DefectResult_with_position calculateDefectPosition(
    const std::optional<std::vector<SensorData>>& csvData,
    const DefectResult& defect,
    int imageHeight = 1024
);

/**
 * @brief 将缺陷结果合并到SQLite数据库（基础版本）
 * @param results 缺陷结果列表
 * @param db_folder 数据库目录
 * @details 创建或覆盖result.db，将结果按自然排序插入数据库
 */
void merge_results_to_db(
    std::vector<DefectResult>& results,
    const std::string& db_folder
);

/**
 * @brief 将缺陷结果合并到SQLite数据库（带位置信息版本）
 * @param results 带位置信息的缺陷结果列表
 * @param db_folder 数据库目录
 * @details 创建或覆盖result.db，包含Position字段，处理DefectType中的"-"分隔符
 */
void merge_results_to_db(
    std::vector<DefectResult_with_position>& results,
    const std::string& db_folder
);

/**
 * @brief 处理单张图像的完整流程
 * @param img_path 图像文件路径
 * @param railhead_output_path 轨面裁剪输出目录
 * @param stretch_output_path 拉伸分割输出目录
 * @param classify_session ONNX分类模型会话
 * @param detector YOLO检测器
 * @param gd_detector 光带异常检测器
 * @param img_size 输入图像尺寸
 * @param crop_threshold 裁剪阈值
 * @param crop_kernel_size 裁剪核大小
 * @param crop_wide 裁剪宽度
 * @param center_limit 是否启用中心限制
 * @param limit_area 限制区域大小
 * @param stretch_ratio 拉伸比例
 * @param results 缺陷结果输出列表
 * @param results_mutex 结果列表互斥锁
 * @param total_pieces_processed 已处理片段计数
 * @param camera_side 相机标识（"L"或"R"）
 * @details 执行轨面提取、拉伸分割、分类检测的完整流程
 */
void process_single_image(
    const std::string& img_path,
    const std::string& railhead_output_path,
    const std::string& stretch_output_path,
    Ort::Session& classify_session,
    YOLO12Infer& detector,
    InspectionGD::GD_AnomalyDetector& gd_detector,
    int img_size,
    int crop_threshold,
    int crop_kernel_size,
    int crop_wide,
    bool center_limit,
    int limit_area,
    int stretch_ratio,
    std::vector<DefectResult>& results,
    std::mutex& results_mutex,
    std::atomic<int>& total_pieces_processed,
    const std::string& camera_side
);

/**
 * @brief 处理单个相机的所有图像
 * @param cam 相机目录名称
 * @param cam_side 相机显示名称
 * @param folder 根目录
 * @param classify_session ONNX分类模型会话
 * @param detector YOLO检测器
 * @param gd_detector 光带异常检测器
 * @param img_size 输入图像尺寸
 * @param crop_threshold 裁剪阈值
 * @param crop_kernel_size 裁剪核大小
 * @param crop_wide 裁剪宽度
 * @param center_limit 是否启用中心限制
 * @param limit_area 限制区域大小
 * @param stretch_ratio 拉伸比例
 * @param local_results 局部结果列表（线程安全）
 * @param results_mutex 结果列表互斥锁
 * @param total_images_processed 总处理图像计数
 * @param total_pieces_processed 总处理片段计数
 * @param camera_side 相机标识（"L"或"R"）
 * @param skip_FirstAndLastImgs_or_not 是否跳过首尾图像
 * @param mark_over_or_not 标记完成标志（暂未使用）
 * @details 多线程处理相机目录下所有图像，支持跳过首尾图像，显示处理进度
 */
void process_camera_images(
    const std::string& cam,
    const std::string& cam_side,
    const std::string& folder,
    Ort::Session& classify_session,
    YOLO12Infer& detector,
    InspectionGD::GD_AnomalyDetector& gd_detector,
    int img_size,
    int crop_threshold,
    int crop_kernel_size,
    int crop_wide,
    bool center_limit,
    int limit_area,
    int stretch_ratio,
    std::vector<DefectResult>& local_results,
    std::mutex& results_mutex,
    std::atomic<int>& total_images_processed,
    std::atomic<int>& total_pieces_processed,
    const std::string& camera_side,
    bool skip_FirstAndLastImgs_or_not,
    bool mark_over_or_not
);
//Add Inspection_v0.3.0.h_n+7
void collectInspectionFolders(
    const fs::path& base_path,
    std::vector<std::string>& folder_list,
    const std::regex& target_regex,
    const std::regex& parent_regex,
    int max_depth = 2
);

#endif // INSPECTION_H