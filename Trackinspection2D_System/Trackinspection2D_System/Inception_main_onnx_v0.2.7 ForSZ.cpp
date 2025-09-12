#include <iostream>
#include <filesystem>
#include <unordered_set> 
#include <vector>
#include <future>
#include <string>
#include <regex>
#include <map>
#include <set>
#include <fstream>
#include <mutex> 
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <onnxruntime_cxx_api.h>
#include "sqlite_loader.h"
#include "xml_loader.h"
#include "csv_loader.h"
#include "InceptionDLLForNB.h"
#include "InceptionUtilsForNB.h"
#include <nlohmann/json.hpp>

#ifndef SQLITE_TRANSIENT
#define SQLITE_TRANSIENT ((void(*)(void*)) -1)
#endif

using namespace std;
namespace fs = std::filesystem;

// 全局SQLite加载器
SQLiteLoader g_sqliteLoader;
// 常量定义
#define IMG_SIZE 256
#define MAX_THREADS 8
std::mutex cout_mutex;

bool Doublemutex = true;
bool TestModel = true;


std::optional<std::vector<SensorData>> csvSensorData(const std::string& filename) {
    std::vector<SensorData> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return std::nullopt; // 相当于返回 None
    }

    std::string line;
    // 跳过标题行（第一行）
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        SensorData record;

        // 解析每个字段
        std::getline(ss, token, '\t');
        record.index = std::stoi(token);

        std::getline(ss, token, '\t');
        record.timestamp = token;

        std::getline(ss, token, '\t');
        record.speed1 = std::stod(token);

        std::getline(ss, token, '\t');
        record.value1 = std::stoi(token);

        std::getline(ss, token, '\t');
        record.speed2 = std::stod(token);

        std::getline(ss, token, '\t');
        record.value2 = std::stoi(token);

        data.push_back(record);
    }

    file.close();
    return data;
}

// 单图像处理流程，全部用InceptionDLL和InceptionUtils封装
void process_single_image(
    const std::string& img_path,
    const std::string& railhead_output_path,
    const std::string& stretch_output_path,
    Ort::Session& classify_session,
    YOLO12Infer& detector,
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
    const std::string& camera_side)
{
    // 1. 切分轨面
    if (!fs::exists(img_path)) {
        std::cerr << "图像文件不存在: " << img_path << std::endl;
        return;
    }
    cv::Mat cropped = InceptionDLL::CropRailhead(img_path, crop_threshold, crop_kernel_size, crop_wide, center_limit, limit_area);
    if (cropped.empty()) return;
    // 检查cropped图像尺寸
    if (cropped.empty() || !cropped.data || cropped.rows <= 0 || cropped.cols <= 0) {
        std::cerr << "无效的裁剪图像: " << img_path << std::endl;
        return;
    }
    std::string cropped_name = fs::path(img_path).filename().string();
    std::string cropped_path = railhead_output_path + "/" + cropped_name;
    fs::create_directories(railhead_output_path);
    InceptionUtils::imwrite_unicode(cropped_path, cropped);

    // 2. 拉伸与分割
    std::vector<std::string> stretch_piece_paths = InceptionDLL::StretchAndSplit(
        cropped, cropped_name, true, stretch_output_path, stretch_ratio);

    // 3. 
    for (const auto& out_path : stretch_piece_paths) {
        // std::cout << "正在推理...." << out_path << endl;
        std::string pred_label = InceptionDLL::ClassifyImage(classify_session, out_path, img_size, out_path);
        std::string result;

        if (TestModel) {
            cout << out_path << "被ClassifyImage识别为" << pred_label<< endl;
        }

        if (pred_label == "DK" || pred_label == "CS" || pred_label == "GF" || pred_label == "HF") {
            if (TestModel == true) {
                cout << out_path << "被识别为DK,执行DetectImage进程" << endl;
            }
            std::string detection_result = InceptionDLL::DetectImage(detector, out_path, out_path);
            result = detection_result;
            if (TestModel == true) {
                cout << "返回的结果是：" << result;
            }
        }
        else {
            nlohmann::json j_result = nlohmann::json::array({ { {"class_name", pred_label} } });
            result = j_result.dump();
        }
        // 统计处理的图像片段数
        total_pieces_processed++;
        // 解析result字符串为json
        try {
            nlohmann::json j;
            // 兼容单个/多个结果
            if (result.front() == '[') {
                j = nlohmann::json::parse(result);
            }
            else {
                j = nlohmann::json::array({ nlohmann::json::parse(result) });
            }
            for (const auto& item : j) {
                if (TestModel == true) {
                    if (item.contains("class_name") && item["class_name"] == "ZC") {
                        std::cout << "Skipping item with class_name ZC" << std::endl;
                        continue;
                    }
                }
                // 快速调整导入DB中的类别
                // 跳过ZC
                if (item.contains("class_name") && item["class_name"] == "ZC") continue;
                // 跳过YC
                if (item.contains("class_name") && item["class_name"] == "YC") continue;
                // 跳过光带不均
                // if (item.contains("class_name") && item["class_name"] == "GD") continue;
                // 跳过波磨
                //  if (item.contains("class_name") && item["class_name"] == "BM") continue;


                DefectResult dr;
                dr.DefectType = item.value("class_name", "");
                if (item.contains("class_name") && item["class_name"].is_string()) {
                    dr.DefectType = item["class_name"].get<std::string>();
                }
                else {
                    dr.DefectType = "UNKNOWN";  // 或其他默认值
                }
                dr.Camera = camera_side;
                dr.ImageName = fs::path(out_path).filename().string();
                if (item.contains("bbox") && item["bbox"].is_array() && item["bbox"].size() == 4) {
                    dr.X = item["bbox"][0];
                    dr.Y = item["bbox"][1];
                    dr.W = item["bbox"][2];
                    dr.H = item["bbox"][3];
                }
                dr.Confidence = item.value("confidence", 0.0f);
                dr.Area = item.value("area", 0.0f);
                if (item.contains("contours")) {
                    dr.Points = item["contours"].dump();
                }
                dr.PointsArea = item.value("area_contour", 0.0f);

                std::lock_guard<std::mutex> lock(results_mutex);
                results.push_back(dr);
            }
        }
        catch (const std::exception& e) {
            DefectResult dr;
            dr.DefectType = pred_label;
            dr.Camera = camera_side;
            cout << "出错的 result 内容: " << result << endl;
            cout << out_path << "解析失败   :\n" << pred_label <<"\n"<< e.what() << endl;
            //解析失败，写入最基本信息
            dr.ImageName = fs::path(out_path).filename().string();
            std::lock_guard<std::mutex> lock(results_mutex);
            results.push_back(dr);
            continue;
        }

    }
}
std::string GbkToUtf8(const std::string& gbkStr) {
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
bool is_valid_utf8(const std::string& str) {
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
void merge_results_to_db(
    std::vector<DefectResult>& results,
    const std::string& db_folder) {

    //// 按照 ImageName 排序
    //std::sort(results.begin(), results.end(), [](const DefectResult& a, const DefectResult& b) {
    //    return a.ImageName < b.ImageName;
    //    });
    // 使用自然排序算法，使得数字按数值大小排序
    std::sort(results.begin(), results.end(), [](const DefectResult& a, const DefectResult& b) {
        // 提取文件名（不包括扩展名）
        auto getNameWithoutExt = [](const std::string& filename) {
            size_t pos = filename.find_last_of('.');
            return (pos != std::string::npos) ? filename.substr(0, pos) : filename;
            };
        std::string filenameA = getNameWithoutExt(a.ImageName);
        std::string filenameB = getNameWithoutExt(b.ImageName);
        // 查找数字部分
        std::regex numRegex("\\d+");
        std::smatch matchA, matchB;
        // 如果两个文件名都包含数字，按数字大小排序
        if (std::regex_search(filenameA, matchA, numRegex) &&
            std::regex_search(filenameB, matchB, numRegex)) {
            int numA = std::stoi(matchA[0]);
            int numB = std::stoi(matchB[0]);
            if (numA != numB) {
                return numA < numB;
            }
        }
        // 如果数字相同或没有数字，按原始文件名排序
        return a.ImageName < b.ImageName;
        });

    std::string db_path = db_folder + R"(\result.db)";
    if (std::filesystem::exists(db_path)) {
        std::filesystem::remove(db_path);
    }
    if (!g_sqliteLoader.isLoaded()) return;
    sqlite3* db;
    char* err_msg = nullptr;
    int rc = g_sqliteLoader.sqlite3_open_fn(db_path.c_str(), &db);
    if (rc != 0) {
        g_sqliteLoader.sqlite3_close_fn(db);
        return;
    }
    const char* create_sql = R"(
        CREATE TABLE IF NOT EXISTS result (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            DefectType TEXT,
            Camera TEXT,
            ImageName TEXT,
            X REAL,
            Y REAL,
            H REAL,
            W REAL,
            Confidence REAL,
            Area REAL,
            Points TEXT,
            PointsArea REAL
        )
    )";
    g_sqliteLoader.sqlite3_exec_fn(db, create_sql, nullptr, nullptr, &err_msg);

    const char* insert_sql = R"(
        INSERT INTO result (
            DefectType, Camera, ImageName, X, Y, H, W, Confidence, Area, Points, PointsArea
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt;
    rc = g_sqliteLoader.sqlite3_prepare_v2_fn(db, insert_sql, -1, &stmt, nullptr);
    for (const auto& dr : results) {


        // DefectType
        if (!dr.DefectType.empty()) {
            if (dr.DefectType == "ZC") {
                // 跳过 DefectType 为 "ZC" 的结果
                continue;
            }
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, 1, dr.DefectType.c_str(), -1, SQLITE_TRANSIENT);
        }
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 1);
        // Camera
        if (!dr.Camera.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, 2, dr.Camera.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 2);
        // ImageName
        if (!dr.ImageName.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, 3, dr.ImageName.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 3);
        // X
        if (dr.X != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, 4, dr.X);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 4);
        // Y
        if (dr.Y != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, 5, dr.Y);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 5);
        // H
        if (dr.H != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, 6, dr.H);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 6);
        // W
        if (dr.W != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, 7, dr.W);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 7);
        // Confidence
        if (dr.Confidence != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, 8, dr.Confidence);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 8);
        // Area
        if (dr.Area != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, 9, dr.Area);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 9);
        // Points
        if (!dr.Points.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, 10, dr.Points.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 10);
        // PointsArea
        if (dr.PointsArea > 0.0f && dr.DefectType == "DK") {
            //cout << "PointsArea:" << dr.PointsArea << endl;
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, 11, dr.PointsArea);
            
        }
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 11);

        g_sqliteLoader.sqlite3_step_fn(stmt);
        g_sqliteLoader.sqlite3_reset_fn(stmt);
        g_sqliteLoader.sqlite3_clear_bindings_fn(stmt);
    }
    g_sqliteLoader.sqlite3_finalize_fn(stmt);
    g_sqliteLoader.sqlite3_close_fn(db);
    std::cout << "预测结果已合并并保存到: " << db_path << std::endl;
}

std::string format_duration(const std::chrono::seconds& duration) {
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    auto seconds = duration - minutes;
    return std::to_string(minutes.count()) + " 分 " + std::to_string(seconds.count()) + " 秒";
}
// 将相机处理封装为函数
void process_camera_images(const std::string& cam,
    const std::string& folder,
    Ort::Session& classify_session,
    YOLO12Infer& detector,
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
    bool mark_over_or_not) {

    std::string cam_side = (cam == "左相机") ? "L" : "R";
    std::string cam_folder = folder + "//" + cam;
    std::string railhead_output_path = folder + "//" + cam + "_railhead";
    std::string stretch_output_path = folder + "//" + cam + "_railhead_stretch";
    std::string csv_path = folder + "//" + "IMAQ_" + cam + ".csv";
    bool csv_file_ready = false;
    auto result = csvSensorData(csv_path);

    if (result) {
        csv_file_ready = true;
    }
    else {
        csv_file_ready = false;
        std::cout << "未能加载" << cam << "csv文件" << std::endl;
    }

    if (fs::is_directory(cam_folder)) {
        fs::create_directories(railhead_output_path);
        fs::create_directories(stretch_output_path);

        std::vector<std::string> image_files;
        for (const auto& entry : fs::directory_iterator(cam_folder)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                    image_files.push_back(entry.path().string());
            }
        }

        if (image_files.empty()) {
            std::cout << "\r\n未检测到" << cam << "图像数据，检查" << cam << "是否异常。" << std::endl;
            return;
        }


        //当前代码导致左相机数据丢失的原因是：process_camera_images 内部重新定义了 results，没有使用外部传入的 local_results，导致主线程无法收集左相机的结果。
        //删除 process_camera_images 内部的以下两行
        //std::vector<DefectResult> results;
        //std::mutex results_mutex;
        int idx = 0, total = static_cast<int>(image_files.size());
        std::atomic<int> finished_count{ 0 };

        auto cam_start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        const size_t max_threads = MAX_THREADS;

        // 0.2.7Update 添加跳过首尾图片检测逻辑
        // 预先计算需要跳过的文件
        std::unordered_set<std::string> files_to_skip;
        if (skip_FirstAndLastImgs_or_not && !image_files.empty()) {
            bool found_special = false;
            for (const auto& img_path : image_files) {
                std::filesystem::path img_path_obj(img_path);
                std::string filename = img_path_obj.filename().string();
                if (filename == "0000000.jpg" || filename == "000000.jpg" || filename == "00000.jpg" || filename == "0000.jpg") {
                    files_to_skip.insert(img_path);
                    found_special = true;
                }
            }
            if (found_special && !image_files.empty()) {
                files_to_skip.insert(image_files.back());
            }
        }

        for (const auto& img_path : image_files) {
            threads.emplace_back([&, img_path]() {
                // 检测是否跳过该文件（首尾）
                if (skip_FirstAndLastImgs_or_not) {
                    if (files_to_skip.find(img_path) != files_to_skip.end()) {
                        // 跳过文件，但更新计数和进度
                        finished_count++;
                        total_images_processed++;

                        std::lock_guard<std::mutex> lock(cout_mutex);
                        int percent = finished_count * 100 / total;
                        std::cout << "\r[" << cam_folder << "] 处理进度: " << percent << "% (" << finished_count << "/" << total << ")" << std::flush;
                        return;
                    }
                }
                process_single_image(
                    img_path,
                    railhead_output_path,
                    stretch_output_path,
                    classify_session,
                    detector,
                    img_size,
                    crop_threshold,
                    5,
                    crop_wide,
                    center_limit,
                    limit_area,
                    stretch_ratio,
                    local_results,
                    results_mutex,
                    total_pieces_processed,
                    cam_side
                );
               /* void process_single_image(
                    const std::string & img_path,
                    const std::string & railhead_output_path,
                    const std::string & stretch_output_path,
                    Ort::Session & classify_session,
                    YOLO12Infer & detector,
                    int img_size,
                    int crop_threshold,
                    int crop_kernel_size,
                    int crop_wide,
                    bool center_limit,
                    int limit_area,
                    int stretch_ratio,
                    std::vector<DefectResult>&results,
                    std::mutex & results_mutex,
                    int& total_pieces_processed,
                    const std::string & camera_side*/
                finished_count++;
                total_images_processed++;

                std::lock_guard<std::mutex> lock(cout_mutex);
                int percent = finished_count * 100 / total;
                std::cout << "\r[" << cam_folder << "] 处理进度: " << percent << "% (" << finished_count << "/" << total << ")" << std::flush;
                });

            if (threads.size() >= max_threads) {
                for (auto& t : threads) t.join();
                threads.clear();
            }
        }

        for (auto& t : threads) t.join();
        std::cout << "\r[" << cam_folder << "] 处理进度: " << 100 << "% (" << finished_count << "/" << total << ")" << std::flush;
        std::cout << std::endl;
        //if (!local_results.empty()) {
        //    cout << "process_camera_images local_results 不为空++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        //}
        //else {
        //    cout << "process_camera_images local_results 为空++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        //}
        auto cam_end_time = std::chrono::high_resolution_clock::now();
        auto cam_duration = std::chrono::duration_cast<std::chrono::seconds>(cam_end_time - cam_start_time);

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "处理 " << cam << " 图像 " << total << " 张，耗时 "
                << format_duration(cam_duration) << std::endl;
        }

        //merge_results_to_db(local_results, folder);
    }
}
int main(int argc, char* argv[]) {
    try {
        const std::string xml_path = "C:\\DataBase2D\\setting For SZ.xml";
        
        std::string img2D_path = XmlLoader::get_value_from_xml(xml_path, "2DDataSetPath", "D://2DImage");
  
        std::string database_root = XmlLoader::get_value_from_xml(xml_path, "2DDataBasePath", "C://DataBase2D");
        // v2.7.0Update 是否启用首末图像跳过
        bool skip_FirstAndLastImgs_or_not = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "skip_FirstAndLastImgs_or_not", "true"));
        // 是否创建over标签文件
        bool mark_over_or_not = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "mark_over_or_not", "true"));
        // 是否使用gpu进行运算
        bool use_gpu = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "use_gpu_or_not", "false"));
        // 是否启动数据收集
        bool data_collect_or_not = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "data_collect_or_not", "false"));

        // 从 XML 文件中获取 CROP_WIDE 和 CROP_THRESHOLD 的值
        int CROP_WIDE = std::stoi(XmlLoader::get_value_from_xml(xml_path, "CROP_WIDE", "660"));
        int CROP_THRESHOLD = std::stoi(XmlLoader::get_value_from_xml(xml_path, "CROP_THRESHOLD", "100"));

        bool center_limit = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "center_limit", "false"));
        int limit_area = std::stoi(XmlLoader::get_value_from_xml(xml_path, "CROP_THRESHOLD", "250"));
        // 从 XML 文件中获取拉伸比stretch_ratio的值
        int STRETCH_RATIO = std::stoi(XmlLoader::get_value_from_xml(xml_path, "STRETCH_RATIO", "2"));
        // ===== 分类检测模块参数 =====
        // 分类器尺寸
        int CLASSIFER_IMG_SIZE = std::stoi(XmlLoader::get_value_from_xml(xml_path, "CLASSIFIER_SIZE", std::to_string(IMG_SIZE)));
        // 分类器置信度阈值
        int CLASSIFER_CONFIDENCE_THR = std::stof(XmlLoader::get_value_from_xml(xml_path, "CLASSIFIER_CONFIDENCE", "0.5f"));
        // ===== 目标检测模块参数 =====
        // 检测器尺寸
        int DETECTOR_IMG_SIZE = std::stoi(XmlLoader::get_value_from_xml(xml_path, "DETECTOR_SIZE", "512"));
        // 检测器置信度阈值
        float DETECTOR_CONFIDENCE_THR = std::stof(XmlLoader::get_value_from_xml(xml_path, "DETECTOR_CONFIDENCE", "0.5f"));
        float DETECTOR_IOU_THR = std::stof(XmlLoader::get_value_from_xml(xml_path, "DETECTOR_IOU", "0.65f"));

        // 禁用OpenCV日志输出
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
        // 解析命令行参数
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-img2D_path" && i + 1 < argc) img2D_path = argv[++i];
            else if (arg == "-database_root" && i + 1 < argc) database_root = argv[++i];
        }

        std::string classify_model_name = XmlLoader::get_value_from_xml(xml_path, "ClassificationModel", "C1.onnx");
        std::string detect_model_name = XmlLoader::get_value_from_xml(xml_path, "DetectionModel", "D1.onnx");

        std::string model_path_str = database_root + "//weights//" + classify_model_name;
        std::wstring wmodel_path = std::wstring(model_path_str.begin(), model_path_str.end());
        std::string detect_model_path_str = database_root + "//weights//" + detect_model_name;
        std::string detect_model_path = std::string(detect_model_path_str.begin(), detect_model_path_str.end());

        Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetLogSeverityLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL);
        
        if (use_gpu) {
#ifdef _WIN32
            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
#endif
        }
        std::cout << "加载检测模型..." << std::endl;
        Ort::Session classify_session(env, wmodel_path.c_str(), session_options);


        YOLO12Infer detector(detect_model_path, cv::Size(DETECTOR_IMG_SIZE, DETECTOR_IMG_SIZE), DETECTOR_CONFIDENCE_THR, DETECTOR_IOU_THR,use_gpu);

        std::cout << "加载模型完成" << std::endl;

        // 路径校验
        if (!fs::exists(img2D_path) || !fs::is_directory(img2D_path)) {
            std::cerr << "2D图像路径异常，请联系开发人员" << std::endl;
            return 0;
        }
        std::vector<std::string> Inspction_folder; // 待检测的文件夹队列
        std::regex folder_regex(R"(((WP|WN)\d+|Fake|Test|2D)+_\d{4}Y\d{2}M\d{2}D\d{2}h\d{2}m\d{2}s)"); 
        //++++===== 文件夹非空非已检测 ====++++
        for (const auto& entry : fs::directory_iterator(img2D_path)) {
            if (entry.is_directory()) {
                std::string folder_name = entry.path().filename().string();
                if (std::regex_match(folder_name, folder_regex)) {
                    if (InceptionUtils::is_over_file_exist(entry.path().string())) {
                        // std::cout << folder << " 已检测，跳过。" << std::endl;
                        continue;
                    }
                    Inspction_folder.push_back(entry.path().string());
                }
            }
        }
        //++++===== 待检测文件夹处理主循环 ====++++
        for (const auto& folder : Inspction_folder) {
            std::cout << "\r\n开始处理线路: " << folder << std::endl;
            if (!fs::is_directory(folder)) continue;
            if (InceptionUtils::is_over_file_exist(folder)) {
                // std::cout << folder << " 已检测，跳过。" << std::endl;
                continue;
            }

            // 记录线路区间处理的开始时间
            auto folder_start_time = std::chrono::high_resolution_clock::now();
            std::atomic<int> total_images_processed(0);  // 用于记录线路区间处理的图像总数
            std::atomic<int> total_pieces_processed(0);  // 用于记录线路区间处理的图像片段总数

            std::vector<DefectResult> results;
            if (!Doublemutex) {
                //++++===== 待检测文件夹左右相机处理（单线程）====++++
                for (const auto& cam : { "左相机", "右相机" }) {
                    std::string cam_side = (std::string(cam) == "左相机") ? "L" : "R";
                    std::string cam_folder = folder + "//" + cam;
                    std::string railhead_output_path = folder + "//" + cam + "_railhead";
                    std::string stretch_output_path = folder + "//" + cam + "_railhead_stretch";
                    std::string csv_path = folder + "//" + "IMAQ_" + cam + ".csv";
                    bool csv_file_ready = false;
                    auto result = csvSensorData(csv_path);

                    if (result) {
                        csv_file_ready = true;
                    }
                    else {
                        csv_file_ready = false;
                        std::cout << "未能加载" << cam << "csv文件" << std::endl;
                    }
                    if (fs::is_directory(cam_folder)) {
                        fs::create_directories(railhead_output_path);
                        fs::create_directories(stretch_output_path);

                        std::vector<std::string> image_files;
                        for (const auto& entry : fs::directory_iterator(cam_folder)) {
                            if (entry.is_regular_file()) {
                                std::string ext = entry.path().extension().string();
                                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                                    image_files.push_back(entry.path().string());
                            }
                        }
                        if (image_files.empty()) {
                            std::cout << "\r\n未检测到" << cam << "图像数据，检查" << cam << "是否异常。" << std::endl;
                            continue;
                        }


                        std::mutex results_mutex;
                        int idx = 0, total = static_cast<int>(image_files.size());
                        std::atomic<int> finished_count{ 0 };

                        // 添加相机处理的开始时间
                        auto cam_start_time = std::chrono::high_resolution_clock::now();

                        // 设置多线程
                        std::vector<std::thread> threads;
                        const size_t max_threads = MAX_THREADS;

                        // 启动工作线程
                        for (const auto& img_path : image_files) {
                            threads.emplace_back([&, img_path]() {
                                process_single_image(
                                    img_path,
                                    railhead_output_path,
                                    stretch_output_path,
                                    classify_session,
                                    detector,
                                    CLASSIFER_IMG_SIZE,
                                    CROP_THRESHOLD,
                                    5,
                                    CROP_WIDE,
                                    center_limit,
                                    limit_area,
                                    STRETCH_RATIO,
                                    results,
                                    results_mutex,
                                    total_pieces_processed,
                                    cam_side
                                );
                                finished_count++;
                                total_images_processed++;
                                {
                                    std::lock_guard<std::mutex> lock(cout_mutex);
                                    int percent = finished_count * 100 / total;
                                    std::cout << "\r[" << cam_folder << "] 顺序处理进度: " << percent << "% (" << finished_count << "/" << total << ")" << std::flush;
                                }
                                });

                            if (threads.size() >= max_threads) {
                                for (auto& t : threads) t.join();
                                threads.clear();
                            }
                        }
                        std::mutex cout_mutex; // 全局互斥锁
                        for (auto& t : threads) t.join();
                        // 停止进度显示线程
                        std::cout << "\r[" << cam_folder << "] 顺序处理进度: " << 100 << "% (" << finished_count << "/" << total << ")" << std::flush;
                        std::cout << std::endl;

                        // 计算相机处理时间
                        auto cam_end_time = std::chrono::high_resolution_clock::now();
                        auto cam_duration = std::chrono::duration_cast<std::chrono::seconds>(cam_end_time - cam_start_time);

                        {
                            std::lock_guard<std::mutex> lock(cout_mutex);
                            std::cout << "处理 " << cam << " 图像 " << total << " 张，耗时 "
                                << format_duration(cam_duration) << std::endl;
                        }

                        // 合并结果到数据库
                        merge_results_to_db(results, folder);
                    }
                }
            }
            else {
                //++++===== 待检测文件夹左右相机处理（双线程）====++++

            // 主处理逻辑
                std::vector<std::future<void>> futures;
                std::atomic<int> global_total_images_processed{ 0 };
                std::atomic<int> global_total_pieces_processed{ 0 };
                std::vector<DefectResult> global_results;
                std::mutex global_results_mutex;
                std::vector<DefectResult> global_results_left;
                std::vector<DefectResult> global_results_right;
                std::mutex global_results_mutex_left;
                std::mutex global_results_mutex_right;

                // 启动左右相机并行处理
                for (const auto& cam : { "左相机", "右相机" }) {
                    futures.push_back(std::async(std::launch::async, [&, cam]() {
                        std::vector<DefectResult> local_results;
                        std::mutex local_results_mutex;
                        std::string cam_side = (cam == "左相机") ? "L" : "R";
                        process_camera_images(
                            cam,
                            folder,
                            classify_session,
                            detector,
                            CLASSIFER_IMG_SIZE,
                            CROP_THRESHOLD,
                            5,
                            CROP_WIDE,
                            center_limit,
                            limit_area,
                            STRETCH_RATIO,
                            local_results, // 使用局部结果集
                            local_results_mutex,
                            global_total_images_processed,
                            global_total_pieces_processed,
                            cam_side,
                            skip_FirstAndLastImgs_or_not,
                            mark_over_or_not
                        );
                        std::lock_guard<std::mutex> lock(global_results_mutex);
                        if (TestModel) {
                            if (!local_results.empty()) {
                                cout << "local_results 不为空++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
                            }
                            else {
                                cout << "local_results 为空++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
                            }
                        }

                        global_results.insert(global_results.end(), local_results.begin(), local_results.end());
                        }));
                }

                // 等待所有相机处理完成
                for (auto& future : futures) {
                    future.get();
                }
                if (TestModel) {
                    if (!global_results.empty()) {
                        cout << "global_results 不为空++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
                    }
                    else {
                        cout << "global_results 为空++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
                    }
                }
                merge_results_to_db(global_results, folder);


            }
            // 计算整个线路处理的总时间
            auto folder_end_time = std::chrono::high_resolution_clock::now();
            auto folder_duration = std::chrono::duration_cast<std::chrono::seconds>(folder_end_time - folder_start_time);

            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "\r\n线路: " << folder << "处理完毕" << std::endl;
            if (total_images_processed > 0) {
                std::cout << "原始图像总数: " << total_images_processed << " 张" << std::endl;
            }
            if (total_pieces_processed > 0) {
                std::cout << "拉伸后处理片段总数: " << total_pieces_processed << " 张" << std::endl;
            }
            std::cout << "总耗时: " << format_duration(folder_duration) << std::endl;
            std::cout << "结果保存路径: " << folder << R"(\result.db)" << std::endl;
            if (mark_over_or_not) {
                InceptionUtils::mark_folder_over(folder);
            }
        }

        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "\r\n当前所有线路检查完成，异常检测进程关闭..." << std::endl;
        std::cout << "All current line anomaly detection are completed, the program is on standby..." << std::endl;
        return 0;
    }catch (const std::exception& e) {
        std::cerr << "程序异常终止: " << e.what() << std::endl;
        return 3;
    }
    catch (...) {
        std::cerr << "程序发生未知异常，异常终止。" << std::endl;
        return 3;
    }
}