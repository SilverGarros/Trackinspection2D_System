#include <iostream>
#include <filesystem>
#include <vector>
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
#include "InceptionDLL.h"
#include "InceptionUtils.h"
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

bool TestModel = false;

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
    int& total_pieces_processed,
    const std::string& camera_side)
{
    // 1. 切分轨面
    
    cv::Mat cropped = InceptionDLL::CropRailhead(img_path, crop_threshold, crop_kernel_size, crop_wide, center_limit, limit_area);
    if (cropped.empty()) return;
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
        std::string result = "['class_name': '" + pred_label + "']";

        if (TestModel) {
            cout << out_path << "被ClassifyImage识别为" << pred_label<< endl;
        }

        if (pred_label == "DK" || pred_label == "CS" || pred_label == "GF") {
            if (TestModel == true) {
                cout << out_path << "被识别为DK,执行DetectImage进程" << endl;
            }
            //const std::string& temp_path = 
            std::string detection_result = InceptionDLL::DetectImage(detector, out_path, out_path);
            result = detection_result;
            if (TestModel == true) {
                cout << "返回的结果是：" << result;
            }
        }
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

                // 跳过ZC
                if (item.contains("class_name") && item["class_name"] == "ZC") continue;
                DefectResult dr;
                dr.DefectType = item.value("class_name", "");
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
            // 解析失败，写入最基本信息
            DefectResult dr;
            dr.DefectType = pred_label;
            dr.Camera = camera_side;
            dr.ImageName = fs::path(out_path).filename().string();
            std::lock_guard<std::mutex> lock(results_mutex);
            results.push_back(dr);
        }
        // 统计处理的图像片段数
        total_pieces_processed++;
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
    const std::string& db_folder,
    const std::string& side) {

    // 按照 ImageName 排序
    std::sort(results.begin(), results.end(), [](const DefectResult& a, const DefectResult& b) {
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
        if (dr.PointsArea != -1){
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

int main(int argc, char* argv[]) {
    try {
        const std::string xml_path = "C:\\DataBase2D\\setting.xml";
        
        std::string img2D_path = XmlLoader::get_value_from_xml(xml_path, "2DDataSetPath", "D://2DImage");
        std::string model_type = "C1";
        int img_size = IMG_SIZE;
        std::string database_root = XmlLoader::get_value_from_xml(xml_path, "2DDataBasePath", "C://DataBase2D");
        bool mark_over_or_not = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "mark_over_or_not", "false"));
        // 从 XML 文件中获取 CROP_WIDE 和 CROP_THRESHOLD 的值
        int CROP_WIDE = std::stoi(XmlLoader::get_value_from_xml(xml_path, "CROP_WIDE", "660"));
        int CROP_THRESHOLD = std::stoi(XmlLoader::get_value_from_xml(xml_path, "CROP_THRESHOLD", "100"));

        bool center_limit = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "center_limit", "false"));
        int limit_area = std::stoi(XmlLoader::get_value_from_xml(xml_path, "CROP_THRESHOLD", "250"));

        // 禁用OpenCV日志输出
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
        // 解析命令行参数
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-img2D_path" && i + 1 < argc) img2D_path = argv[++i];
            else if (arg == "-model_type" && i + 1 < argc) model_type = argv[++i];
            else if (arg == "-database_root" && i + 1 < argc) database_root = argv[++i];
        }

        std::string model_name = XmlLoader::get_value_from_xml(xml_path, "ClassificationModel", "C1.onnx");
        std::string detect_model_name = XmlLoader::get_value_from_xml(xml_path, "DetectionModel", "D1.onnx");

        std::string model_path_str = database_root + "//weights//" + model_name;
        std::wstring wmodel_path = std::wstring(model_path_str.begin(), model_path_str.end());
        std::string detect_model_path_str = database_root + "//weights//" + detect_model_name;
        std::string detect_model_path = std::string(detect_model_path_str.begin(), detect_model_path_str.end());

        Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetLogSeverityLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL);
        
#ifdef _WIN32
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
#endif  
        std::cout << "加载检测模型..." << std::endl;
        Ort::Session session(env, wmodel_path.c_str(), session_options);
        std::cout << "加载模型完成" << std::endl;

        YOLO12Infer detector(detect_model_path, cv::Size(512, 512), 0.8f, 0.45f);

        // 路径校验
        if (!fs::exists(img2D_path) || !fs::is_directory(img2D_path)) {
            std::cerr << "2D图像路径异常，请联系开发人员" << std::endl;
            return 0;
        }
        std::vector<std::string> Inspction_folder;
        std::regex folder_regex(R"(((WP|WN)\d|Fake)+_\d{4}Y\d{2}M\d{2}D\d{2}h\d{2}m\d{2}s)"); 
        for (const auto& entry : fs::directory_iterator(img2D_path)) {
            if (entry.is_directory()) {
                std::string folder_name = entry.path().filename().string();
                if (std::regex_match(folder_name, folder_regex)) {
                    Inspction_folder.push_back(entry.path().string());
                }
            }
        }
        for (const auto& folder : Inspction_folder) {
            std::cout << "\r\n开始处理线路: " << folder << std::endl;
            if (!fs::is_directory(folder)) continue;
            if (InceptionUtils::is_over_file_exist(folder)) {
                // std::cout << folder << " 已检测，跳过。" << std::endl;
                continue;
            }

            // 添加线路处理的开始时间
            auto folder_start_time = std::chrono::high_resolution_clock::now();
            int total_images_processed = 0;  // 跟踪处理的图像总数
            int total_pieces_processed = 0;  // 跟踪处理的图像片段总数

            std::vector<DefectResult> results;
            for (const auto& cam : { "左相机", "右相机" }) {
                std::string cam_side = (std::string(cam) == "左相机") ? "L" : "R";
                std::string cam_folder = folder + "//" + cam;
                std::string railhead_output_path = folder + "//" + cam + "_railhead";
                std::string stretch_output_path = folder + "//" + cam + "_railhead_stretch";
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
                                session,
                                detector,
                                img_size,
                                CROP_THRESHOLD,
                                5,
                                CROP_WIDE,
                                center_limit,
                                limit_area,
                                2,
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
                    merge_results_to_db(results, folder, cam_side);
                }
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
        }

        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "\r\n当前所有线路检查完成，异常检测进程待机中..." << std::endl;
        std::cout << "All current line anomaly detection are completed, the program is on standby..." << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "程序异常终止: " << e.what() << std::endl;
        return 3;
    }
    catch (...) {
        std::cerr << "程序发生未知异常，异常终止。" << std::endl;
        return 3;
    }
}