#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <regex>
#include <map>
#include <set>
#include <thread>
#include <future>
#include <fstream>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "sqlite_loader.h"
#include "detector.h"

using namespace std;
namespace fs = std::filesystem;

// 全局SQLite加载器
SQLiteLoader g_sqliteLoader;
// 常量定义
#define IMG_SIZE 256
#define CROP_WIDE 512
#define CROP_THRESHOLD 28
#define DK_THRESHOLD 70

// 类别映射表
static const std::unordered_map<int, std::string> classes_label_map = {
    {0, "YC"}, {1, "DK"}, {2, "BM"}, {3, "HF"},
    {4, "CS"}, {5, "ZC"}, {6, "GF"}, {7, "GD"}
};

// 提取图片编号
std::string extract_image_num(const std::string& filename) {
    std::regex re(R"((\d+)_\d+of\d+\.\w+)");
    std::smatch match;
    if (std::regex_match(filename, match, re)) {
        return match[1];
    }
    return "";
}

// 检查over文件
bool is_over_file_exist(const std::string& folder) {
    return fs::exists(folder + "/over");
}

// 标记over
void mark_folder_over(const std::string& folder) {
    std::ofstream file(folder + "/over");
    file.close();
}

// ONNX分类预测
std::string class_predict_onnx(Ort::Session& session, const std::string& img_path, int img_size) {
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        cout << "class_predict_onnx 获取的的图像" << img_path << "为空" << "?" << std::endl;
        return "110 Unknown";
    }
    cv::resize(img, img, cv::Size(img_size, img_size));
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
        int max_index = 0;
        float max_value = output_data[0];
        for (size_t i = 1; i < output_count; ++i) {
            if (output_data[i] > max_value) {
                max_value = output_data[i];
                max_index = static_cast<int>(i);
            }
        }
        auto it = classes_label_map.find(max_index);
        return (it != classes_label_map.end()) ? it->second : "111 Unknown";
    }
    catch (...) {
        return "112 Unknown";
    }
}

// 检测预测
std::vector<DetectionResult> detection_onnx(YOLO12Infer& detector, const std::string& img_path) {
    try {
        return detector.predict(img_path, false, true, true, true);
    }
    catch (const std::exception& e) {
        std::cerr << "[detection_onnx] std::exception: " << e.what() << " | img_path: " << img_path << std::endl;
        return {};
    }
    catch (...) {
        std::cerr << "[detection_onnx] 未知异常 | img_path: " << img_path << std::endl;
        return {};
    }
}

// 轨面高亮区域自动裁剪（单张图像）
cv::Mat railhead_crop_highlight_center_area(
    const cv::Mat& img, int threshold, int kernel_size, int crop_wide)
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
        if (std::abs(crod_m - img_center) > 50)
            crod_m = img_center;
    }

    int x1 = max(0, crod_m - crop_wide / 2);
    int x2 = min(img.cols, crod_m + crop_wide / 2);
    int y1 = 0, y2 = img.rows;
    if (x2 <= x1 || y2 <= y1) {
        x1 = max(0, img_center - crop_wide / 2);
        x2 = min(img.cols, img_center + crop_wide / 2);
    }
    return img(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
}

// 轨面高亮区域自动裁剪
void process_images_in_folder_thread(
    const std::string& input_folder,
    const std::string& output_folder,
    int threshold,
    int kernel_size,
    int crop_wide,
    int max_workers = 8)

{
    namespace fs = std::filesystem;
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                image_files.push_back(entry.path().string());
        }
    }
    std::mutex io_mutex;
    std::atomic<int> progress_count(0);
    int total = static_cast<int>(image_files.size());
    std::vector<std::future<void>> futures;
    for (const auto& img_path : image_files) {
        futures.push_back(std::async(std::launch::async, [&, img_path]() {
            cv::Mat img = cv::imread(img_path);
            if (img.empty()) return;
            cv::Mat cropped = railhead_crop_highlight_center_area(img, threshold, kernel_size, crop_wide);
            std::string out_path = output_folder + "/" + fs::path(img_path).filename().string();
            {
                std::lock_guard<std::mutex> lock(io_mutex);
                fs::create_directories(output_folder);
                cv::imwrite(out_path, cropped);
            }
            int current = ++progress_count;
            if (total > 0) {
                int percent = current * 100 / total;
                std::lock_guard<std::mutex> lock(io_mutex);
                std::cout << "\r[" << input_folder << "] 轨面切分执行中...进度: " << percent << "% (" << current << "/" << total << ")" << std::flush;
            }
            }));
        if (futures.size() >= (size_t)max_workers) {
            for (auto& f : futures) f.get();
            futures.clear();
        }
    }
    for (auto& f : futures) f.get();
    if (total > 0) std::cout << std::endl;
}

// 轨面拉伸
void folder_image_vertical_stretch_and_split(
    const std::string& input_folder,
    const std::string& output_folder,
    int stretch_ratio = 2,
    int max_workers = 32)
{
    namespace fs = std::filesystem;
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                image_files.push_back(entry.path().string());
        }
    }
    std::mutex io_mutex;
    std::atomic<int> progress_count(0);
    int total = static_cast<int>(image_files.size());
    std::vector<std::future<void>> futures;
    for (const auto& img_path : image_files) {
        futures.push_back(std::async(std::launch::async, [&, img_path]() {
            cv::Mat img = cv::imread(img_path);
            if (img.empty()) return;
            int orig_h = img.rows, orig_w = img.cols;
            int new_h = orig_h * stretch_ratio;
            cv::Mat stretched;
            cv::resize(img, stretched, cv::Size(orig_w, new_h), 0, 0, cv::INTER_LINEAR);

            int count = new_h / orig_h;
            int rem = new_h % orig_h;
            std::string base = fs::path(img_path).stem().string();
            std::string ext = fs::path(img_path).extension().string();
            for (int i = 0; i < count; ++i) {
                cv::Mat piece = stretched.rowRange(i * orig_h, (i + 1) * orig_h);
                std::string out_name = base + "_" + std::to_string(count + (rem ? 1 : 0)) + "of" + std::to_string(i + 1) + ext;
                std::string out_path = output_folder + "/" + out_name;
                {
                    std::lock_guard<std::mutex> lock(io_mutex);
                    fs::create_directories(output_folder);
                    cv::imwrite(out_path, piece);
                }
            }
            if (rem) {
                cv::Mat piece = stretched.rowRange(count * orig_h, new_h);
                std::string out_name = base + "_" + std::to_string(count + 1) + "of" + std::to_string(count + 1) + ext;
                std::string out_path = output_folder + "/" + out_name;
                {
                    std::lock_guard<std::mutex> lock(io_mutex);
                    fs::create_directories(output_folder);
                    cv::imwrite(out_path, piece);
                }
            }
            int current = ++progress_count;
            if (total > 0) {
                int percent = current * 100 / total;
                std::lock_guard<std::mutex> lock(io_mutex);
                std::cout << "\r[" << input_folder << "] 轨面拉伸执行中...进度: " << percent << "% (" << current << "/" << total << ")" << std::flush;
            }
            }));
        if (futures.size() >= (size_t)max_workers) {
            for (auto& f : futures) f.get();
            futures.clear();
        }
    }
    for (auto& f : futures) f.get();
    if (total > 0) std::cout << std::endl;
}

void batch_predict_and_merge(
    const std::string& folder_path,
    Ort::Session& classify_session,
    YOLO12Infer& detector,
    int img_size,
    const std::string& db_folder,
    const std::string& side,
    int max_workers = 8)
{
    namespace fs = std::filesystem;
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                image_files.push_back(entry.path().string());
        }
    }
    std::map<std::string, std::vector<std::string>> results;
    std::mutex results_mutex;
    std::atomic<int> progress_count(0);
    int total = static_cast<int>(image_files.size());
    std::vector<std::future<void>> futures;
    for (const auto& img_path : image_files) {
        futures.push_back(std::async(std::launch::async, [&]() {
            std::string pred_label = class_predict_onnx(classify_session, img_path, img_size);
            // std::cout << pred_label << std::endl;
            if (pred_label == "DK") {
                //std::cout <<"执行detection_onnx" << std::endl;
                auto det_results = detection_onnx(detector, img_path);
                if (!det_results.empty() && det_results.size() == 1 && det_results[0].class_name == "ZC") {
                    pred_label = "ZC";
                }
                else if (!det_results.empty()) {
                    pred_label = detection_results_to_string(det_results);
                }
                else {
                    pred_label = "ZC";
                }
            }
            std::string img_name = fs::path(img_path).filename().string();
            std::string image_num = extract_image_num(img_name);
            if (!image_num.empty()) {
                std::lock_guard<std::mutex> lock(results_mutex);
                results[image_num].push_back(pred_label);
            }
            int current = ++progress_count;
            if (total > 0) {
                int percent = current * 100 / total;
                std::lock_guard<std::mutex> lock(results_mutex);
                std::cout << "\r[" << folder_path << "] 预测执行中...进度: " << percent << "% (" << current << "/" << total << ")" << std::flush;
            }
            }));
        if (futures.size() >= (size_t)max_workers) {
            for (auto& f : futures) f.get();
            futures.clear();
        }
    }
    for (auto& f : futures) {
        try {
            f.get();
        }
        catch (const std::exception& e) {
            std::cerr << "batch_predict_and_merge线程异常: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "batch_predict_and_merge线程发生未知异常" << std::endl;
        }
    }
    if (total > 0) std::cout << std::endl;

    // 合并结果到数据库
    std::string db_path = db_folder + "/result.db";
    if (!g_sqliteLoader.isLoaded()) return;
    sqlite3* db;
    char* err_msg = nullptr;
    int rc = g_sqliteLoader.sqlite3_open_fn(db_path.c_str(), &db);
    if (rc != 0) {
        g_sqliteLoader.sqlite3_close_fn(db);
        return;
    }
    const char* create_sql = "CREATE TABLE IF NOT EXISTS result (ImageNum TEXT PRIMARY KEY, L TEXT, R TEXT)";
    g_sqliteLoader.sqlite3_exec_fn(db, create_sql, nullptr, nullptr, &err_msg);
    std::map<std::string, std::pair<std::string, std::string>> old_results;
    const char* select_sql = "SELECT ImageNum, L, R FROM result";
    sqlite3_stmt* stmt;
    rc = g_sqliteLoader.sqlite3_prepare_v2_fn(db, select_sql, -1, &stmt, nullptr);
    if (rc == 0) {
        while (g_sqliteLoader.sqlite3_step_fn(stmt) == 100) {
            std::string img_num = (const char*)g_sqliteLoader.sqlite3_column_text_fn(stmt, 0);
            std::string l_val = g_sqliteLoader.sqlite3_column_text_fn(stmt, 1) ? (const char*)g_sqliteLoader.sqlite3_column_text_fn(stmt, 1) : "";
            std::string r_val = g_sqliteLoader.sqlite3_column_text_fn(stmt, 2) ? (const char*)g_sqliteLoader.sqlite3_column_text_fn(stmt, 2) : "";
            old_results[img_num] = { l_val, r_val };
        }
    }
    g_sqliteLoader.sqlite3_finalize_fn(stmt);

    g_sqliteLoader.sqlite3_exec_fn(db, "BEGIN TRANSACTION", nullptr, nullptr, &err_msg);
    std::set<std::string> all_image_nums;
    for (const auto& kv : old_results) all_image_nums.insert(kv.first);
    for (const auto& kv : results) all_image_nums.insert(kv.first);

    const char* insert_sql = "INSERT OR REPLACE INTO result (ImageNum, L, R) VALUES (?, ?, ?)";
    rc = g_sqliteLoader.sqlite3_prepare_v2_fn(db, insert_sql, -1, &stmt, nullptr);

    for (const auto& image_num : all_image_nums) {
        std::string l_val = old_results[image_num].first;
        std::string r_val = old_results[image_num].second;
        if (side == "L" && results.find(image_num) != results.end()) {
            std::stringstream ss;
            for (size_t i = 0; i < results[image_num].size(); i++) {
                if (i > 0) ss << ",";
                ss << results[image_num][i];
            }
            l_val = ss.str();
        }
        if (side == "R" && results.find(image_num) != results.end()) {
            std::stringstream ss;
            for (size_t i = 0; i < results[image_num].size(); i++) {
                if (i > 0) ss << ",";
                ss << results[image_num][i];
            }
            r_val = ss.str();
        }
        g_sqliteLoader.sqlite3_bind_text_fn(stmt, 1, image_num.c_str(), -1, nullptr);
        g_sqliteLoader.sqlite3_bind_text_fn(stmt, 2, l_val.c_str(), -1, nullptr);
        g_sqliteLoader.sqlite3_bind_text_fn(stmt, 3, r_val.c_str(), -1, nullptr);

        rc = g_sqliteLoader.sqlite3_step_fn(stmt);
        g_sqliteLoader.sqlite3_reset_fn(stmt);
        g_sqliteLoader.sqlite3_clear_bindings_fn(stmt);
    }
    g_sqliteLoader.sqlite3_finalize_fn(stmt);
    g_sqliteLoader.sqlite3_exec_fn(db, "COMMIT", nullptr, nullptr, &err_msg);
    g_sqliteLoader.sqlite3_close_fn(db);

    std::cout << "预测结果已合并并保存到: " << db_path << std::endl;
    // mark_folder_over(folder_path); // 如需自动标记可取消注释
}

// 单张图片全流程处理（切分、拉伸、分类检测）
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
    int stretch_ratio,
    std::map<std::string, std::vector<std::string>>& results,
    std::mutex& results_mutex)
{
    namespace fs = std::filesystem;
    // 1. 切分轨面
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) return;
    cv::Mat cropped = railhead_crop_highlight_center_area(img, crop_threshold, crop_kernel_size, crop_wide);
    std::string cropped_name = fs::path(img_path).filename().string();
    std::string cropped_path = railhead_output_path + "/" + cropped_name;
    fs::create_directories(railhead_output_path);
    cv::imwrite(cropped_path, cropped);

    // 2. 拉伸
    int orig_h = cropped.rows, orig_w = cropped.cols;
    int new_h = orig_h * stretch_ratio;
    cv::Mat stretched;
    cv::resize(cropped, stretched, cv::Size(orig_w, new_h), 0, 0, cv::INTER_LINEAR);

    int count = new_h / orig_h;
    int rem = new_h % orig_h;
    std::string base = fs::path(cropped_name).stem().string();
    std::string ext = fs::path(cropped_name).extension().string();
    std::vector<std::string> stretch_piece_paths;
    fs::create_directories(stretch_output_path);
    for (int i = 0; i < count; ++i) {
        cv::Mat piece = stretched.rowRange(i * orig_h, (i + 1) * orig_h);
        std::string out_name = base + "_" + std::to_string(count + (rem ? 1 : 0)) + "of" + std::to_string(i + 1) + ext;
        std::string out_path = stretch_output_path + "/" + out_name;
        cv::imwrite(out_path, piece);
        stretch_piece_paths.push_back(out_path);
    }
    if (rem) {
        cv::Mat piece = stretched.rowRange(count * orig_h, new_h);
        std::string out_name = base + "_" + std::to_string(count + 1) + "of" + std::to_string(count + 1) + ext;
        std::string out_path = stretch_output_path + "/" + out_name;
        cv::imwrite(out_path, piece);
        stretch_piece_paths.push_back(out_path);
    }

    // 3. 分类检测
    for (const auto& piece_path : stretch_piece_paths) {
        std::string pred_label = class_predict_onnx(classify_session, piece_path, img_size);
        if (pred_label == "DK") {
            auto det_results = detection_onnx(detector, piece_path);
            if (!det_results.empty() && det_results.size() == 1 && det_results[0].class_name == "ZC") {
                pred_label = "ZC";
            }
            else if (!det_results.empty()) {
                std::stringstream ss;
                for (size_t i = 0; i < det_results.size(); ++i) {
                    if (i > 0) ss << "|";
                    ss << det_results[i].class_name;
                }
                pred_label = ss.str();
            }
            else {
                pred_label = "ZC";
            }
        }
        std::string img_name = fs::path(piece_path).filename().string();
        std::string image_num = extract_image_num(img_name);
        if (!image_num.empty()) {
            std::lock_guard<std::mutex> lock(results_mutex);
            results[image_num].push_back(pred_label);
        }
    }
}
void merge_results_to_db(
    const std::map<std::string, std::vector<std::string>>& results,
    const std::string& db_folder,
    const std::string& side)
{
    std::string db_path = db_folder + "/result.db";
    if (!g_sqliteLoader.isLoaded()) return;
    sqlite3* db;
    char* err_msg = nullptr;
    int rc = g_sqliteLoader.sqlite3_open_fn(db_path.c_str(), &db);
    if (rc != 0) {
        g_sqliteLoader.sqlite3_close_fn(db);
        return;
    }
    const char* create_sql = "CREATE TABLE IF NOT EXISTS result (ImageNum TEXT PRIMARY KEY, L TEXT, R TEXT)";
    g_sqliteLoader.sqlite3_exec_fn(db, create_sql, nullptr, nullptr, &err_msg);
    std::map<std::string, std::pair<std::string, std::string>> old_results;
    const char* select_sql = "SELECT ImageNum, L, R FROM result";
    sqlite3_stmt* stmt;
    rc = g_sqliteLoader.sqlite3_prepare_v2_fn(db, select_sql, -1, &stmt, nullptr);
    if (rc == 0) {
        while (g_sqliteLoader.sqlite3_step_fn(stmt) == 100) {
            std::string img_num = (const char*)g_sqliteLoader.sqlite3_column_text_fn(stmt, 0);
            std::string l_val = g_sqliteLoader.sqlite3_column_text_fn(stmt, 1) ? (const char*)g_sqliteLoader.sqlite3_column_text_fn(stmt, 1) : "";
            std::string r_val = g_sqliteLoader.sqlite3_column_text_fn(stmt, 2) ? (const char*)g_sqliteLoader.sqlite3_column_text_fn(stmt, 2) : "";
            old_results[img_num] = { l_val, r_val };
        }
    }
    g_sqliteLoader.sqlite3_finalize_fn(stmt);

    g_sqliteLoader.sqlite3_exec_fn(db, "BEGIN TRANSACTION", nullptr, nullptr, &err_msg);
    std::set<std::string> all_image_nums;
    for (const auto& kv : old_results) all_image_nums.insert(kv.first);
    for (const auto& kv : results) all_image_nums.insert(kv.first);

    const char* insert_sql = "INSERT OR REPLACE INTO result (ImageNum, L, R) VALUES (?, ?, ?)";
    rc = g_sqliteLoader.sqlite3_prepare_v2_fn(db, insert_sql, -1, &stmt, nullptr);

    for (const auto& image_num : all_image_nums) {
        std::string l_val = old_results[image_num].first;
        std::string r_val = old_results[image_num].second;
        if (side == "L" && results.find(image_num) != results.end()) {
            std::stringstream ss;
            for (size_t i = 0; i < results.at(image_num).size(); i++) {
                if (i > 0) ss << ",";
                ss << results.at(image_num)[i];
            }
            l_val = ss.str();
        }
        if (side == "R" && results.find(image_num) != results.end()) {
            std::stringstream ss;
            for (size_t i = 0; i < results.at(image_num).size(); i++) {
                if (i > 0) ss << ",";
                ss << results.at(image_num)[i];
            }
            r_val = ss.str();
        }
        g_sqliteLoader.sqlite3_bind_text_fn(stmt, 1, image_num.c_str(), -1, nullptr);
        g_sqliteLoader.sqlite3_bind_text_fn(stmt, 2, l_val.c_str(), -1, nullptr);
        g_sqliteLoader.sqlite3_bind_text_fn(stmt, 3, r_val.c_str(), -1, nullptr);

        rc = g_sqliteLoader.sqlite3_step_fn(stmt);
        g_sqliteLoader.sqlite3_reset_fn(stmt);
        g_sqliteLoader.sqlite3_clear_bindings_fn(stmt);
    }
    g_sqliteLoader.sqlite3_finalize_fn(stmt);
    g_sqliteLoader.sqlite3_exec_fn(db, "COMMIT", nullptr, nullptr, &err_msg);
    g_sqliteLoader.sqlite3_close_fn(db);

    std::cout << "预测结果已合并并保存到: " << db_path << std::endl;
}
int old_main(int argc, char* argv[]) {
    try {
        std::string img2D_path = "I://2DImage";
        std::string model_type = "C1";
        int img_size = IMG_SIZE;
        std::string database_root = "D://LuHang_System//Trackinspection2D_System//DataBase"; // 默认DataBase根目录

        // 解析命令行参数
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-img2D_path" && i + 1 < argc) img2D_path = argv[++i];
            else if (arg == "-model_type" && i + 1 < argc) model_type = argv[++i];
            else if (arg == "-database_root" && i + 1 < argc) database_root = argv[++i];
        }

        // 自动拼接DataBase下的所有路径
        std::string model_path = database_root + "/weights/C1.onnx";
        std::wstring wmodel_path = std::wstring(model_path.begin(), model_path.end());
        std::string detect_model_path_str = database_root + "/weights/D1.onnx";
        std::wstring detect_model_path = std::wstring(detect_model_path_str.begin(), detect_model_path_str.end());

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        // 设置ONNX Runtime日志级别为INFO（1），以便输出更详细的节点分配和Memcpy警告信息
        session_options.SetLogSeverityLevel(4); // 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL

#ifdef _WIN32
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
#endif
        Ort::Session session(env, wmodel_path.c_str(), session_options);
        std::cout << "加载模型完成" << std::endl;
        std::cout << "加载检测模型..." << std::endl;
        YOLO12Infer detector(detect_model_path, cv::Size(512, 512), 0.8f, 0.45f);

        // 路径校验
        if (!fs::exists(img2D_path) || !fs::is_directory(img2D_path)) {
            std::cerr << "2D图像路径异常，请联系开发人员" << std::endl;
            return 0;
        }
        std::vector<std::string> Inspction_folder;
        std::regex folder_regex(R"(2D_\d{14})");
        for (const auto& entry : fs::directory_iterator(img2D_path)) {
            if (entry.is_directory()) {
                std::string folder_name = entry.path().filename().string();
                if (std::regex_match(folder_name, folder_regex)) {
                    Inspction_folder.push_back(entry.path().string());
                }
            }
        }
        for (const auto& folder : Inspction_folder) {
            std::cout << "开始处理文件夹: " << folder << std::endl;
            if (!fs::is_directory(folder)) continue;
            if (is_over_file_exist(folder)) {
                std::cout << folder << " 已检测，跳过。" << std::endl;
                continue;
            }
            for (const auto& cam : { "左相机", "右相机" }) {
                std::string cam_side = (std::string(cam) == "左相机") ? "L" : "R";
                std::string cam_folder = folder + "/" + cam;
                std::string railhead_output_path = folder + "/" + cam + "_railhead";
                std::string stretch_output_path = folder + "/" + cam + "_railhead_stretch";
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
                    std::map<std::string, std::vector<std::string>> results;
                    std::mutex results_mutex;
                    int idx = 0, total = static_cast<int>(image_files.size());
                    for (const auto& img_path : image_files) {
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
                            2,
                            results,
                            results_mutex
                        );
                        idx++;
                        if (total > 0) {
                            int percent = idx * 100 / total;
                            std::cout << "\r[" << cam_folder << "] 顺序处理进度: " << percent << "% (" << idx << "/" << total << ")" << std::flush;
                        }
                    }
                    std::cout << std::endl;
                    // 合并结果到数据库
                    merge_results_to_db(results, folder, cam_side);
                }
            }
            //mark_folder_over(folder);
            //std::cout << folder << " 检测完成，已标记 over 。" << std::endl;
        }
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