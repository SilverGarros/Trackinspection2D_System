#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif
#endif

#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <type_traits>
#include <condition_variable>
#include <atomic>
#include <future>
#include <sstream>
#include <random>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <codecvt>
#include <locale>
#include <set>
#include <chrono>
#include "sqlite_loader.h"
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#include <process.h>
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include <nlohmann/json.hpp>
#include "InceptionTRTDLL.h"
#include "InceptionUtils.h"
#include "InceptionConfig.h"
#include <regex>

namespace fs = std::filesystem;

// 处理结果结构体
struct ProcessingResult {
    std::string image_path;
    std::vector<InceptionResult> inception_results;
    std::chrono::milliseconds processing_time;
    bool success;
    std::string error_message;
    int classification_count;
    int detection_count;
    int total_detected_objects;
};

// 帮助信息
void print_help(const char* program_name) {
    std::cout << "InceptionTRT 多线程图像处理工具" << std::endl;
    std::cout << "使用方法: " << program_name << " [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  -c, --config <file>     配置文件路径 (默认: config.json)" << std::endl;
    std::cout << "  -d, --dir <path>        测试图像目录路径" << std::endl;
    std::cout << "  -t, --threads <num>     工作线程数 (默认: 4)" << std::endl;
    std::cout << "  -n, --num <count>       最大测试图像数量 (默认: 50)" << std::endl;
    std::cout << "  -h, --help              显示此帮助信息" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << program_name << " --config config.json" << std::endl;
    std::cout << "  " << program_name << " --dir \"G:\\test_images\" --threads 8" << std::endl;
    std::cout << "  " << program_name << " -d \"C:\\images\" -t 4 -n 100" << std::endl;
}

// 收集图像文件
std::vector<std::string> collect_image_files(const std::string& image_dir, int max_count) {
    std::vector<std::string> image_files;

    if (!fs::exists(image_dir) || !fs::is_directory(image_dir)) {
        std::cerr << "错误: 图像目录不存在: " << image_dir << std::endl;
        return image_files;
    }

    std::set<std::string> image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
    };

    int collected_count = 0;
    //for (const auto& entry : fs::recursive_directory_iterator(image_dir)) {
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (entry.is_regular_file() && collected_count < max_count) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

            if (image_extensions.find(extension) != image_extensions.end()) {
                image_files.push_back(entry.path().string());
                collected_count++;
            }
        }
    }

    return image_files;
}

// 统计检测对象数量
int count_detected_objects(const std::vector<InceptionResult>& results) {
    int total_objects = 0;
    for (const auto& result : results) {
        if (result.result_type == InceptionResult::DETECTION) {
            total_objects += static_cast<int>(result.detectionresults.size());
        }
    }
    return total_objects;
}

// 多线程 InceptionTRT 理论性能测试函数
void run_inception_multithread_processing(const InceptionConfig& config) {
    std::cout << "=== InceptionTRT 多线程图像处理 ===" << std::endl;
    std::cout << "分类引擎: " << config.classification_engine << std::endl;
    std::cout << "检测引擎: " << config.detection_engine << std::endl;
    std::cout << "测试目录: " << config.test_image_dir << std::endl;
    std::cout << "工作线程: " << config.max_Inspetion_workers << std::endl;
    std::cout << "最大图像: " << config.max_test_images << std::endl;
    std::cout << "拉伸比例: " << config.stretch_ratio << std::endl;
    std::cout << "=================================" << std::endl;

    // 收集图像文件
    std::vector<std::string> image_files = collect_image_files(config.test_image_dir, config.max_test_images);
    if (image_files.empty()) {
        std::cout << "未找到图像文件" << std::endl;
        return;
    }
    std::cout << "找到 " << image_files.size() << " 个图像文件" << std::endl;

    // 多线程设置
    std::queue<std::string> image_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> processing_done(false);

    // 结果存储
    std::vector<ProcessingResult> results;
    std::mutex results_mutex;

    // 统计数据
    std::atomic<int> processed_count(0);
    std::atomic<int> success_count(0);
    std::atomic<int> failed_count(0);
    std::atomic<int> total_classifications(0);
    std::atomic<int> total_detections(0);
    std::atomic<int> total_objects_detected(0);

    // 初始化队列
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        for (const auto& image_file : image_files) {
            image_queue.push(image_file);
        }
    }

    auto start_time = std::chrono::steady_clock::now();
    std::cout << "\n=== 开始多线程处理 ===" << std::endl;

    // 工作线程函数
    auto worker_function = [&](int thread_id) {
        try {

            InceptionTRT processor(
                config.classification_engine,
                config.detection_engine,
                cv::Size(256, 256),  // 分类输入尺寸
                cv::Size(512, 512),  // 检测输入尺寸
                config.confidence_thresh,
                config.iou_thresh,
                config.stretch_ratio
            );
            std::cout << "[线程" << thread_id << "] InceptionTRT 初始化完成" << std::endl;

            while (!processing_done) {
                std::string image_path;

                // 获取任务
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_cv.wait(lock, [&]() { return !image_queue.empty() || processing_done; });

                    if (processing_done && image_queue.empty()) {
                        break;
                    }

                    if (!image_queue.empty()) {
                        image_path = image_queue.front();
                        image_queue.pop();
                    }
                    else {
                        continue;
                    }
                }

                ProcessingResult proc_result;
                proc_result.image_path = image_path;
                proc_result.success = false;
                proc_result.classification_count = 0;
                proc_result.detection_count = 0;
                proc_result.total_detected_objects = 0;

                try {
                    auto start = std::chrono::steady_clock::now();

                    // 执行 InceptionTRT 处理
                    std::vector<InceptionResult> inception_results = processor.process(
                        image_path,
                        config.CROP_WIDE,
                        config.CROP_THRESHOLD,
                        config.CENTER_LIMIT,
                        config.LIMIT_AREA,
                        "temp_thread_" + std::to_string(thread_id),
                        false  // 不保存拉伸图像
                    );

                    auto end = std::chrono::steady_clock::now();
                    proc_result.processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    proc_result.inception_results = std::move(inception_results);
                    proc_result.success = true;

                    // 统计结果
                    for (const auto& result : proc_result.inception_results) {
                        if (result.result_type == InceptionResult::CLASSIFICATION) {
                            proc_result.classification_count++;
                            total_classifications++;
                        }
                        else if (result.result_type == InceptionResult::DETECTION) {
                            proc_result.detection_count++;
                            total_detections++;
                            proc_result.total_detected_objects += static_cast<int>(result.detectionresults.size());
                        }
                    }

                    total_objects_detected += proc_result.total_detected_objects;
                    success_count++;

                    // 输出进度
                    int current = processed_count.fetch_add(1) + 1;
                    if (current % 5 == 0 || current <= 10) {
                        std::cout << "[线程" << thread_id << "] " << current << "/" << image_files.size()
                            << " - " << fs::path(image_path).filename().string()
                            << " (" << proc_result.processing_time.count() << "ms, "
                            << "片段:" << proc_result.inception_results.size()
                            << ", 分类:" << proc_result.classification_count
                            << ", 检测:" << proc_result.detection_count
                            << ", 对象:" << proc_result.total_detected_objects << ")" << std::endl;
                    }

                }
                catch (const std::exception& ex) {
                    proc_result.error_message = ex.what();
                    failed_count++;
                    processed_count++;

                    std::cerr << "[线程" << thread_id << "] 处理失败: "
                        << fs::path(image_path).filename().string()
                        << " - " << ex.what() << std::endl;
                }

                // 保存结果
                {
                    std::lock_guard<std::mutex> lock(results_mutex);
                    results.push_back(std::move(proc_result));
                }
            }

            std::cout << "[线程" << thread_id << "] 处理完成" << std::endl;

        }
        catch (const std::exception& ex) {
            std::cerr << "[线程" << thread_id << "] 初始化失败: " << ex.what() << std::endl;
        }
        };

    // 启动工作线程
    std::vector<std::thread> worker_threads;
    for (int i = 0; i < config.max_Inspetion_workers; ++i) {
        worker_threads.emplace_back(worker_function, i);
    }

    // 监控线程
    std::thread monitor_thread([&]() {
        while (!processing_done) {
            std::this_thread::sleep_for(std::chrono::seconds(3));

            int current_processed = processed_count.load();
            if (current_processed > 0) {
                double progress = (current_processed * 100.0) / image_files.size();
                std::cout << "=== 进度: " << current_processed << "/" << image_files.size()
                    << " (" << std::fixed << std::setprecision(1) << progress << "%) "
                    << "成功:" << success_count.load() << " 失败:" << failed_count.load()
                    << " 分类:" << total_classifications.load() << " 检测:" << total_detections.load()
                    << " 对象:" << total_objects_detected.load() << " ===" << std::endl;
            }
        }
        });

    // 等待处理完成
    while (processed_count.load() < static_cast<int>(image_files.size())) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    processing_done = true;
    queue_cv.notify_all();

    // 等待线程结束
    for (auto& thread : worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }

    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // 计算统计信息
    long long total_inference_time = 0;
    long long min_time = LLONG_MAX, max_time = 0;
    int total_segments = 0;

    {
        std::lock_guard<std::mutex> lock(results_mutex);
        for (const auto& result : results) {
            if (result.success) {
                long long time_ms = result.processing_time.count();
                total_inference_time += time_ms;
                min_time = std::min(min_time, time_ms);
                max_time = std::max(max_time, time_ms);
                total_segments += static_cast<int>(result.inception_results.size());
            }
        }
    }

    // 输出最终统计
    std::cout << "\n=== InceptionTRT 多线程处理完成 ===" << std::endl;
    std::cout << "配置信息:" << std::endl;
    std::cout << "  工作线程数    : " << config.max_Inspetion_workers << std::endl;
    std::cout << "  拉伸比例      : " << config.stretch_ratio << std::endl;
    std::cout << "  置信度阈值    : " << config.confidence_thresh << std::endl;
    std::cout << "  IOU阈值       : " << config.iou_thresh << std::endl;
    std::cout << std::endl;

    std::cout << "处理结果:" << std::endl;
    std::cout << "  总处理图像    : " << processed_count.load() << std::endl;
    std::cout << "  成功处理      : " << success_count.load() << std::endl;
    std::cout << "  失败处理      : " << failed_count.load() << std::endl;
    std::cout << "  总拉伸片段    : " << total_segments << std::endl;
    std::cout << "  分类执行次数  : " << total_classifications.load() << std::endl;
    std::cout << "  检测执行次数  : " << total_detections.load() << std::endl;
    std::cout << "  检测对象总数  : " << total_objects_detected.load() << std::endl;
    std::cout << std::endl;

    std::cout << "性能指标:" << std::endl;
    std::cout << "  总耗时        : " << total_time.count() << "ms" << std::endl;

    if (success_count.load() > 0) {
        std::cout << "  平均处理时间  : " << (total_inference_time / success_count.load()) << "ms/图像" << std::endl;
        std::cout << "  最快处理      : " << min_time << "ms" << std::endl;
        std::cout << "  最慢处理      : " << max_time << "ms" << std::endl;
        std::cout << "  平均吞吐量    : " << std::fixed << std::setprecision(2)
            << (success_count.load() * 1000.0 / total_time.count()) << " 图像/秒" << std::endl;

        if (total_segments > 0) {
            std::cout << "  平均片段数    : " << std::fixed << std::setprecision(1)
                << (total_segments * 1.0 / success_count.load()) << " 片段/图像" << std::endl;
            std::cout << "  分类比例      : " << std::fixed << std::setprecision(1)
                << (total_classifications.load() * 100.0 / total_segments) << "%" << std::endl;
            std::cout << "  检测比例      : " << std::fixed << std::setprecision(1)
                << (total_detections.load() * 100.0 / total_segments) << "%" << std::endl;
        }

        if (total_detections.load() > 0) {
            std::cout << "  平均检测对象  : " << std::fixed << std::setprecision(1)
                << (total_objects_detected.load() * 1.0 / total_detections.load()) << " 对象/检测" << std::endl;
        }
    }

    std::cout << "  成功率        : " << std::fixed << std::setprecision(2)
        << (processed_count.load() > 0 ? (success_count.load() * 100.0 / processed_count.load()) : 0) << "%" << std::endl;

    // 显示部分处理结果示例
    std::cout << "\n=== 处理结果示例 ===" << std::endl;
    {
        std::lock_guard<std::mutex> lock(results_mutex);
        int sample_count = 0;
        for (const auto& result : results) {
            if (result.success && sample_count < 3) {
                std::cout << "\n图像: " << fs::path(result.image_path).filename().string() << std::endl;
                std::cout << "  处理时间: " << result.processing_time.count() << "ms" << std::endl;
                std::cout << "  拉伸片段: " << result.inception_results.size() << std::endl;
                std::cout << "  分类次数: " << result.classification_count << std::endl;
                std::cout << "  检测次数: " << result.detection_count << std::endl;
                std::cout << "  检测对象: " << result.total_detected_objects << std::endl;

                for (size_t i = 0; i < std::min(result.inception_results.size(), size_t(2)); ++i) {
                    const auto& inception_result = result.inception_results[i];
                    std::cout << "    片段" << (i + 1) << ": ";
                    if (inception_result.result_type == InceptionResult::CLASSIFICATION) {
                        std::cout << "分类 - " << inception_result.classificationresult.class_name
                            << " (置信度: " << std::fixed << std::setprecision(3)
                            << inception_result.classificationresult.confidence << ")";
                    }
                    else {
                        std::cout << "检测 - 发现 " << inception_result.detectionresults.size() << " 个目标";
                        for (const auto& detection : inception_result.detectionresults) {
                            std::cout << " [" << detection.class_name << ":"
                                << std::fixed << std::setprecision(2) << detection.confidence << "]";
                        }
                    }
                    std::cout << std::endl;
                }
                sample_count++;
            }
        }
    }
    std::cout << "======================" << std::endl;
}

// === 高性能线程池 ===
class HighPerformanceThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    mutable std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{ false };

public:
    HighPerformanceThreadPool(size_t threads) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
                });
        }
    }

    ~HighPerformanceThreadPool() {
        stop = true;
        condition.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable()) worker.join();
        }
    }

    template<class F>
    auto enqueue(F&& f) -> std::future<std::invoke_result_t<F>> {
        using return_type = std::invoke_result_t<F>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::forward<F>(f)
        );

        std::future<return_type> res = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }

        condition.notify_one();
        return res;
    }

    size_t queue_size() const {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return tasks.size();
    }

    size_t worker_count() const {
        return workers.size();
    }

    bool is_stopped() const {
        return stop.load();
    }
};

void merge_inception_results_to_db(
    const std::vector<InceptionResult>& left_results,
    const std::vector<InceptionResult>& right_results,
    const std::string& db_folder) {

    // 合并左右相机结果
    std::vector<InceptionResult> all_results = left_results;
    all_results.insert(all_results.end(), right_results.begin(), right_results.end());

    // 按照 img_name 排序
    std::sort(all_results.begin(), all_results.end(), [](const InceptionResult& a, const InceptionResult& b) {
        return a.img_name < b.img_name;
        });

    std::string db_path = db_folder + R"(\inception_results.db)";
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
        CREATE TABLE IF NOT EXISTS inception_results (
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
        INSERT INTO inception_results (
            DefectType, Camera, ImageName, X, Y, H, W, Confidence, Area, Points, PointsArea
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt;
    rc = g_sqliteLoader.sqlite3_prepare_v2_fn(db, insert_sql, -1, &stmt, nullptr);

    for (const auto& result : all_results) {
        // 跳过 ClassificationResult 中 class_name 为 "ZC" 的结果
        if (result.result_type == InceptionResult::CLASSIFICATION &&
            result.classificationresult.class_name == "ZC") {
            continue;
        }

        if (result.result_type == InceptionResult::CLASSIFICATION) {
            // 没有 detectionresults，插入一条记录
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, 1, result.classificationresult.class_name.c_str(), -1, SQLITE_TRANSIENT); // DefectType
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 2); // Camera
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, 3, result.img_name.c_str(), -1, SQLITE_TRANSIENT); // ImageName
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 4); // X
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 5); // Y
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 6); // H
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 7); // W
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, 8, result.classificationresult.confidence); // Confidence
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 9); // Area
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 10); // Points
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, 11); // PointsArea

            g_sqliteLoader.sqlite3_step_fn(stmt);
            g_sqliteLoader.sqlite3_reset_fn(stmt);
            g_sqliteLoader.sqlite3_clear_bindings_fn(stmt);
        }
        else if (result.result_type == InceptionResult::DETECTION) {
            // 有 detectionresults，每个 DetectionResult 插入一条记录
            for (const auto& detection : result.detectionresults) {
                g_sqliteLoader.sqlite3_bind_text_fn(stmt, 1, detection.class_name.c_str(), -1, SQLITE_TRANSIENT); // DefectType
                g_sqliteLoader.sqlite3_bind_null_fn(stmt, 2); // Camera
                g_sqliteLoader.sqlite3_bind_text_fn(stmt, 3, result.img_name.c_str(), -1, SQLITE_TRANSIENT); // ImageName
                g_sqliteLoader.sqlite3_bind_double_fn(stmt, 4, detection.bbox.x); // X
                g_sqliteLoader.sqlite3_bind_double_fn(stmt, 5, detection.bbox.y); // Y
                g_sqliteLoader.sqlite3_bind_double_fn(stmt, 6, detection.bbox.height); // H
                g_sqliteLoader.sqlite3_bind_double_fn(stmt, 7, detection.bbox.width); // W
                g_sqliteLoader.sqlite3_bind_double_fn(stmt, 8, detection.confidence); // Confidence
                g_sqliteLoader.sqlite3_bind_double_fn(stmt, 9, detection.area); // Area

                // Points
                if (!detection.contours.empty()) {
                    std::ostringstream points_stream;
                    for (const auto& contour : detection.contours) {
                        for (const auto& point : contour) {
                            points_stream << "(" << point.x << "," << point.y << ") ";
                        }
                    }
                    std::string points = points_stream.str();
                    g_sqliteLoader.sqlite3_bind_text_fn(stmt, 10, points.c_str(), -1, SQLITE_TRANSIENT);
                }
                else {
                    g_sqliteLoader.sqlite3_bind_null_fn(stmt, 10);
                }

                g_sqliteLoader.sqlite3_bind_double_fn(stmt, 11, detection.area_contour); // PointsArea

                g_sqliteLoader.sqlite3_step_fn(stmt);
                g_sqliteLoader.sqlite3_reset_fn(stmt);
                g_sqliteLoader.sqlite3_clear_bindings_fn(stmt);
            }
        }
    }

    g_sqliteLoader.sqlite3_finalize_fn(stmt);
    g_sqliteLoader.sqlite3_close_fn(db);
    std::cout << "Inception 结果已合并并保存到: " << db_path << std::endl;
}
// 主函数
int main() {
    try {
        // 设置 OpenCV 日志等级
#ifdef _WIN32
        // 设置高优先级以获得更好的CPU调度
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

        // 优化内存管理
        HANDLE heap = GetProcessHeap();
        HeapSetInformation(heap, HeapEnableTerminationOnCorruption, nullptr, 0);


        _putenv_s("OPENCV_LOG_LEVEL", "ERROR");

        _putenv_s("CUDA_CACHE_DISABLE", "0");  // 启用CUDA缓存
        _putenv_s("CUDA_FORCE_PTX_JIT", "0");  // 禁用运行时JIT编译
#else
        setenv("OPENCV_LOG_LEVEL", "ERROR", 1);
        setenv("CUDA_CACHE_DISABLE", "0", 1);
        setenv("CUDA_FORCE_PTX_JIT", "0", 1);
#endif
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

        // 优化CUDA设备设置
        cudaSetDevice(0);
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);  // 优化L1缓存
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);  // 优化共享内存


        // 固定配置文件路径
        const std::string config_file = "C:\\DataBase2D\\config.json";
        InceptionConfig config = InceptionConfig::load_from_file(config_file);

        std::cout << "配置文件加载成功: " << config_file << std::endl;

        // 配置验证
        if (!config.validate()) {
            std::cerr << "错误: 配置验证失败" << std::endl;
            return 1;
        }
        // 路径校验
        std::string img2D_path = config.DataSetPath;
        if (!fs::exists(img2D_path) || !fs::is_directory(img2D_path)) {
            std::cerr << "2D图像路径异常，请联系开发人员: " << img2D_path << std::endl;
            return 0;
        }
        //// 启动测试流程
        //run_inception_multithread_processing(config);

        // 初始化 InceptionTRT 处理器
        std::cout << "初始化 InceptionTRT 处理器 测试..." << std::endl;

        // 根据GPU内存和线程数优化处理器数量
        const int optimal_processors = std::min(config.max_workers,16);  // 限制最大数量避免GPU内存不足
        std::cout << "optimal_processors" << optimal_processors<< std::endl;


        std::vector<std::unique_ptr<InceptionTRT>> processor_pool;
        processor_pool.reserve(optimal_processors);
        //InceptionTRT processor(
        //    config.classification_engine,
        //    config.detection_engine,
        //    cv::Size(256, 256),  // 分类输入尺寸
        //    cv::Size(512, 512),  // 检测输入尺寸
        //    config.confidence_thresh,
        //    config.iou_thresh,
        //    config.stretch_ratio
        //);


        // 并行初始化处理器池
        std::vector<std::future<std::unique_ptr<InceptionTRT>>> init_futures;
        for (int i = 0; i < optimal_processors; ++i) {
            init_futures.push_back(std::async(std::launch::async, [&config, i]() {
                return std::make_unique<InceptionTRT>(
                    config.classification_engine,
                    config.detection_engine,
                    cv::Size(256, 256),  // 分类输入尺寸
                    cv::Size(512, 512),  // 检测输入尺寸
                    config.confidence_thresh,
                    config.iou_thresh,
                    config.stretch_ratio
                );
                }));
        }
        // 收集初始化结果
        for (auto& future : init_futures) {
            processor_pool.push_back(future.get());
        }
        std::cout << "成功初始化 " << processor_pool.size() << " 个处理器实例" << std::endl;
        std::cout << "InceptionTRT 处理器初始化测试成功" << std::endl;

        // === 处理器池管理 ===
        std::queue<int> available_processors;
        std::mutex processor_mutex;
        std::condition_variable processor_cv;
        // 初始化可用处理器队列
        for (int i = 0; i < optimal_processors; ++i) {
            available_processors.push(i);
        }
        std::cout << "处理器池队列初始化完成，可用处理器: " << available_processors.size() << " 个" << std::endl;

        // 查找符合规则的文件夹
        std::vector<std::string> inspection_folders;
        std::regex folder_regex(R"(((WP|WN)\d+|Fake)+_\d{4}Y\d{2}M\d{2}D\d{2}h\d{2}m\d{2}s)");
        for (const auto& entry : fs::directory_iterator(img2D_path)) {
            if (entry.is_directory()) {
                std::string folder_name = entry.path().filename().string();
                if (std::regex_match(folder_name, folder_regex)) {
                    inspection_folders.push_back(entry.path().string());
                }
            }
        }
        if (inspection_folders.empty()) {
            std::cout << "未找到符合命名规则的检测文件夹" << std::endl;
            return 0;
        }
        std::cout << "找到 " << inspection_folders.size() << " 个待检测文件夹" << std::endl;



        // 创建高性能线程池
        HighPerformanceThreadPool thread_pool(optimal_processors);  // 增加线程数以提高并发

        // === 性能监控器 ===
        struct PerformanceStats {
            std::atomic<long long> total_processing_time_us{ 0 };
            std::atomic<int> images_processed{ 0 };
            std::atomic<int> pieces_processed{ 0 };
            std::chrono::steady_clock::time_point start_time;

            PerformanceStats() : start_time(std::chrono::steady_clock::now()) {}

            void add_result(long long processing_time_us, int pieces) {
                total_processing_time_us += processing_time_us;
                images_processed++;
                pieces_processed += pieces;
            }

            void print_stats() const {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);

                if (images_processed > 0 && elapsed.count() > 0) {
                    double avg_time = total_processing_time_us.load() / (images_processed.load() * 1000.0);
                    double throughput = images_processed.load() / (double)elapsed.count();

                    std::cout << "=== 实时性能统计 ===" << std::endl;
                    std::cout << "处理图像: " << images_processed << " 张" << std::endl;
                    std::cout << "处理片段: " << pieces_processed << " 个" << std::endl;
                    std::cout << "平均耗时: " << std::fixed << std::setprecision(2) << avg_time << " ms/图" << std::endl;
                    std::cout << "处理速度: " << std::setprecision(1) << throughput << " 图像/秒" << std::endl;
                    std::cout << "=================" << std::endl;
                }
            }
        };

        // === 单行进度显示器 ===
        struct ProgressDisplay {
            std::mutex progress_mutex;
            std::chrono::steady_clock::time_point start_time;
            int left_total = 0, right_total = 0;
            std::atomic<int> left_processed{ 0 }, right_processed{ 0 };
            std::string current_folder;

            ProgressDisplay() : start_time(std::chrono::steady_clock::now()) {}

            void set_totals(int left_count, int right_count, const std::string& folder) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                left_total = left_count;
                right_total = right_count;
                current_folder = fs::path(folder).filename().string();
                left_processed = 0;
                right_processed = 0;
                start_time = std::chrono::steady_clock::now();
            }

            void update_progress(bool is_left, int increment = 1) {
                if (is_left) {
                    left_processed += increment;
                }
                else {
                    right_processed += increment;
                }

                // 每处理50张图像更新一次显示
                int total_processed = left_processed + right_processed;
                if (total_processed % 50 == 0 ||
                    (left_processed >= left_total && right_processed >= right_total)) {
                    display_progress();
                }
            }

            void display_progress() {
                std::lock_guard<std::mutex> lock(progress_mutex);

                int total_images = left_total + right_total;
                int total_processed = left_processed + right_processed;

                if (total_images == 0) return;

                double progress_percent = (total_processed * 100.0) / total_images;

                // 计算处理速度
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
                double speed = elapsed.count() > 0 ? total_processed / (double)elapsed.count() : 0;

                // 估算剩余时间
                int remaining = total_images - total_processed;
                int eta_seconds = speed > 0 ? (int)(remaining / speed) : 0;

                // 创建进度条
                const int bar_width = 10;  // 
                int filled = (int)(progress_percent * bar_width / 100.0);
                std::string progress_bar = "[";
                for (int i = 0; i < bar_width; ++i) {
                    if (i < filled) progress_bar += "=";      // 使用 = 代替 █
                    else progress_bar += " ";                 // 使用空格代替 ░
                }
                progress_bar += "]";

                std::string clear_line(100, ' ');
                std::cout << "\r" << clear_line;  // 先清除整行

                // 单行输出进度信息 - 使用\r回到行首覆盖之前的内容
                std::cout << "\r" << current_folder << " " << progress_bar
                    << " " << std::fixed << std::setprecision(1) << progress_percent << "% "
                    << "(" << total_processed << "/" << total_images << ") "
                    << "左:" << left_processed << "/" << left_total << " "
                    << "右:" << right_processed << "/" << right_total << " "
                    << "速度:" << std::setprecision(1) << speed << "图/s";

                if (eta_seconds > 0 && eta_seconds < 3600) {  // 只显示小于1小时的预估时间
                    int eta_minutes = eta_seconds / 60;
                    eta_seconds %= 60;
                    std::cout << " 剩余:" << eta_minutes << "m" << eta_seconds << "s";
                }

                std::cout << std::flush;// 强制刷新输出缓冲区

                // 如果处理完成，换行
                if (total_processed >= total_images) {
                    std::cout << "\n";
                }
            }

            void force_newline() {
                std::lock_guard<std::mutex> lock(progress_mutex);
                std::cout << "\n";
            }
        };

        PerformanceStats global_stats;
        ProgressDisplay progress_display;
        for (const auto& folder : inspection_folders) {
            std::cout << "\r\n开始处理线路: " << folder << std::endl;

            if (!fs::is_directory(folder)) continue;

            // 检查是否已处理过
            if (InceptionUtils::is_over_file_exist(folder)) {
                std::cout << folder << " 已检测，跳过。" << std::endl;
                continue;
            }
            // 记录线路处理开始时间
            auto folder_start_time = std::chrono::high_resolution_clock::now();
            std::atomic<int> total_images_processed{ 0 };
            std::atomic<int> total_pieces_processed{ 0 };
            std::mutex results_mutex;

            // 预先统计左右相机图像数量
            std::atomic<int> left_count{ 0 }, right_count{ 0 };
            std::vector<std::future<std::pair<int, bool>>> count_futures;
            for (const auto& cam : { "左相机", "右相机" }) {
                count_futures.push_back(thread_pool.enqueue([&, cam]() -> std::pair<int, bool> {
                    std::string cam_folder = folder + "//" + cam;
                    bool is_left = (std::string(cam) == "左相机");

                    if (!fs::is_directory(cam_folder)) {
                        return { 0, is_left };
                    }

                    int count = 0;
                    const std::set<std::string> valid_extensions{ ".jpg", ".jpeg", ".png", ".bmp" };

                    for (const auto& entry : fs::directory_iterator(cam_folder)) {
                        if (entry.is_regular_file()) {
                            std::string ext = entry.path().extension().string();
                            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                            if (valid_extensions.count(ext)) {
                                count++;
                            }
                        }
                    }

                    return { count, is_left };
                    }));
            }
            // 等待统计完成
            for (auto& future : count_futures) {
                auto result = future.get();
                if (result.second) {  // 左相机
                    left_count = result.first;
                }
                else {  // 右相机
                    right_count = result.first;
                }
            }

            // 设置进度显示器
            progress_display.set_totals(left_count, right_count, folder);


            // 定义左右相机结果容器
            std::vector<InceptionResult> left_results;
            std::vector<InceptionResult> right_results;

            // 处理左右相机
            std::vector<std::future<void>> camera_futures;
            for (const auto& cam : { "左相机", "右相机" }) {
                std::string cam_side = (std::string(cam) == "左相机") ? "L" : "R";
                camera_futures.push_back(thread_pool.enqueue([&, cam]() {
                    bool is_left_camera = (std::string(cam) == "左相机");
                    std::string cam_folder = folder + "//" + cam;
                    std::string stretch_output_path = folder + "//" + cam + "_railhead_stretch";

                    if (!fs::is_directory(cam_folder)) {
                        std::cout << "相机文件夹不存在: " << cam_folder << std::endl;
                        return;
                    }

                    // 创建输出目录
                    fs::create_directories(stretch_output_path);

                    // 收集图像文件
                    std::vector<std::string> image_files;
                    image_files.reserve(1000);  // 预分配内存

                    const std::set<std::string> valid_extensions{ ".jpg", ".jpeg", ".png", ".bmp" };

                    for (const auto& entry : fs::directory_iterator(cam_folder)) {
                        if (entry.is_regular_file()) {
                            std::string ext = entry.path().extension().string();
                            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                            if (valid_extensions.count(ext)) {
                                image_files.push_back(entry.path().string());
                            }
                        }
                    }

                    if (image_files.empty()) {
                        std::cout << "\r\n未检测到" << cam << "图像数据" << std::endl;
                        return;
                    }

                    std::cout << "\r\n开始处理 " << cam << " (" << image_files.size() << " 张图像)" << std::endl;

                    // 高性能并行处理图像
                    std::atomic<int> finished_count{ 0 };
                    std::vector<std::future<void>> image_futures;

                    for (const auto& img_path : image_files) {
                        image_futures.push_back(thread_pool.enqueue([&, img_path]() {
                            // 获取处理器
                            int processor_id = -1;
                            {
                                std::unique_lock<std::mutex> lock(processor_mutex);
                                processor_cv.wait(lock, [&] { return !available_processors.empty(); });
                                processor_id = available_processors.front();
                                available_processors.pop();
                            }

                            try {
                                auto start_time = std::chrono::steady_clock::now();

                                // 执行InceptionTRT处理 - 使用处理器池中的实例
                                std::vector<InceptionResult> inception_results =
                                    processor_pool[processor_id]->process(
                                        img_path,
                                        config.CROP_WIDE,
                                        config.CROP_THRESHOLD,
                                        config.CENTER_LIMIT,
                                        config.LIMIT_AREA,
                                        stretch_output_path,
                                        true  // 禁用文件保存以提升性能
                                    );

                                auto end_time = std::chrono::steady_clock::now();
                                auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
                                    end_time - start_time);

                                int image_pieces = static_cast<int>(inception_results.size());
                                {
                                    std::lock_guard<std::mutex> lock(results_mutex);
                                    if (is_left_camera) {
                                        left_results.insert(left_results.end(), inception_results.begin(), inception_results.end());
                                    }
                                    else {
                                        right_results.insert(right_results.end(), inception_results.begin(), inception_results.end());
                                    }
                                }
                                // 更新统计
                                total_images_processed++;
                                total_pieces_processed += image_pieces;
                                global_stats.add_result(processing_time.count(), image_pieces);
                                finished_count++;


                                // 设置进度显示器
                                progress_display.update_progress(is_left_camera);

                            }
                            catch (const std::exception& ex) {
                                // 设置进度显示器
                                progress_display.update_progress(is_left_camera);
                                finished_count++;
                            }

                            // 释放处理器
                            {
                                std::lock_guard<std::mutex> lock(processor_mutex);
                                available_processors.push(processor_id);
                                processor_cv.notify_one();
                            }
                            }));
                    }

                    // 等待当前相机所有图像处理完成
                    for (auto& future : image_futures) {
                        future.wait();
                    }

                    std::cout << "[" << cam << "] 处理完成: " << finished_count
                        << "/" << image_files.size() << " 张图像" << std::endl;
                    }));
            }

            // 等待所有相机处理完成
            for (auto& future : camera_futures) {
                future.wait();
            }

            // 确保进度条显示完成状态
            progress_display.display_progress();
            progress_display.force_newline();

            // 计算整个线路处理时间
            auto folder_end_time = std::chrono::high_resolution_clock::now();
            auto folder_duration = std::chrono::duration_cast<std::chrono::seconds>(
                folder_end_time - folder_start_time);

            std::cout << "\r\n线路: " << folder << " 处理完毕" << std::endl;
            std::cout << "原始图像总数: " << total_images_processed << " 张" << std::endl;
            std::cout << "拉伸片段总数: " << total_pieces_processed << " 个" << std::endl;
            std::cout << "总耗时: " << folder_duration.count() << " 秒" << std::endl;

            if (folder_duration.count() > 0) {
                double folder_throughput = total_images_processed.load() / (double)folder_duration.count();
                std::cout << "线路处理速度: " << std::fixed << std::setprecision(2)
                    << folder_throughput << " 图像/秒" << std::endl;
            }
            merge_inception_results_to_db(left_results, right_results, folder);

            // 标记文件夹处理完成
            InceptionUtils::mark_folder_over(folder);
            std::cout << "处理完成标记已创建" << std::endl;

            // 每处理完一个线路显示总体性能统计
            global_stats.print_stats();
        }

        // 等待所有任务完成
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        std::cout << "\r\n=== 所有检测任务完成 ===" << std::endl;
        global_stats.print_stats();

        // GPU内存清理
        cudaDeviceReset();

    }
    catch (const std::exception& e) {
        std::cerr << "程序异常: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}