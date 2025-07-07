// Windows 特定定义 - 必须在所有其他头文件之前
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN     // 排除很少使用的Windows API
#endif
#ifndef NOMINMAX
#define NOMINMAX                // 防止Windows.h定义min/max宏
#endif
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_            // 防止windows.h包含winsock.h
#endif
#endif

#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
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
#include <opencv2/opencv.hpp>

// Windows特定的头文件 - 按正确顺序包含
#ifdef _WIN32
#include <process.h>
// 在包含winsock2.h之前先包含windows.h（但由于WIN32_LEAN_AND_MEAN，只包含核心API）
#include <windows.h>
// HTTP服务器相关
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

// JSON处理
#include <nlohmann/json.hpp>

// Inception_TRT_DLL
#include "InceptionTRTDLL.h"

namespace fs = std::filesystem;

// 简化的Unicode路径处理工具函数（避免复杂的Windows API调用）
namespace UnicodePathUtils {

#ifdef _WIN32
    // 简化的文件存在性检查
    bool safe_file_exists(const std::string& path) {
        try {
            // 使用C++17 filesystem，它在Windows上通常能处理UTF-8路径
            std::error_code ec;
            bool exists = fs::exists(path, ec);
            return exists && !ec;
        }
        catch (const std::exception&) {
            return false;
        }
    }

    // 规范化路径格式
    std::string normalize_path(const std::string& path) {
        std::string normalized = path;
        // 将反斜杠替换为正斜杠（filesystem可以处理）
        std::replace(normalized.begin(), normalized.end(), '\\', '/');
        return normalized;
    }

#else
    // Linux/Unix 系统的实现
    bool safe_file_exists(const std::string& path) {
        std::error_code ec;
        return fs::exists(path, ec) && !ec;
    }

    std::string normalize_path(const std::string& path) {
        return path;
    }
#endif
}

// 检测任务结构
struct DetectionTask {
    std::string image_path;
    std::string task_id;
    std::chrono::steady_clock::time_point submit_time;
    std::promise<std::string> result_promise;

    DetectionTask() = default;
    DetectionTask(DetectionTask&& other) noexcept
        : image_path(std::move(other.image_path))
        , task_id(std::move(other.task_id))
        , submit_time(other.submit_time)
        , result_promise(std::move(other.result_promise)) {
    }

    DetectionTask& operator=(DetectionTask&& other) noexcept {
        if (this != &other) {
            image_path = std::move(other.image_path);
            task_id = std::move(other.task_id);
            submit_time = other.submit_time;
            result_promise = std::move(other.result_promise);
        }
        return *this;
    }
};

// 配置结构
struct InceptionConfig {
    std::string classification_engine;
    std::string detection_engine;
    std::string trtexec_path;
    std::string web_root;
    int port = 2003;
    int max_workers = 4;

    static InceptionConfig load_from_file(const std::string& config_file) {
        InceptionConfig config;

        if (!fs::exists(config_file)) {
            std::cout << "配置文件不存在，使用默认配置: " << config_file << std::endl;
            // 设置默认值
            config.classification_engine = "C:/DataBase2D/weights/C1.trt";
            config.detection_engine = "C:/DataBase2D/weights/D1.trt";
            config.trtexec_path = "C:/driver/TensorRT-10.12.0.36/bin/trtexec.exe";
            config.web_root = "web";
            config.port = 2003;
            config.max_workers = 4;
            return config;
        }

        try {
            std::ifstream file(config_file);
            nlohmann::json json_config;
            file >> json_config;

            config.classification_engine = json_config.value("classification_engine", "C:/DataBase2D/weights/C1.trt");
            config.detection_engine = json_config.value("detection_engine", "C:/DataBase2D/weights/D1.trt");
            config.trtexec_path = json_config.value("trtexec_path", "C:/driver/TensorRT-10.12.0.36/bin/trtexec.exe");
            config.web_root = json_config.value("web_root", "web");
            config.port = json_config.value("port", 2003);
            config.max_workers = json_config.value("max_workers", 4);

            std::cout << "配置文件加载成功: " << config_file << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "配置文件解析失败: " << e.what() << std::endl;
            std::cerr << "使用默认配置" << std::endl;
            // 设置默认值
            config.classification_engine = "C:/DataBase2D/weights/C1.trt";
            config.detection_engine = "C:/DataBase2D/weights/D1.trt";
            config.trtexec_path = "C:/driver/TensorRT-10.12.0.36/bin/trtexec.exe";
            config.web_root = "web";
            config.port = 2003;
            config.max_workers = 4;
        }

        return config;
    }
};

// ONNX到TensorRT转换器
class ModelConverter {
public:
    static bool convertOnnxToTensorRT(const std::string& onnx_file, const std::string& trt_file,
        const std::string& trtexec_path) {
        if (!fs::exists(onnx_file)) {
            std::cerr << "ONNX文件不存在: " << onnx_file << std::endl;
            return false;
        }

        if (!fs::exists(trtexec_path)) {
            std::cerr << "trtexec.exe路径不存在: " << trtexec_path << std::endl;
            return false;
        }

        std::cout << "正在转换ONNX模型到TensorRT: " << onnx_file << " -> " << trt_file << std::endl;

        // 构建trtexec命令
        std::ostringstream cmd;
        cmd << "\"" << trtexec_path << "\" "
            << "--onnx=" << onnx_file << " "
            << "--saveEngine=" << trt_file << " ";

        std::string command = cmd.str();
        std::cout << "执行命令: " << command << std::endl;

        // 执行转换命令
        int result = std::system(command.c_str());

        if (result != 0) {
            std::cerr << "TensorRT转换失败，返回码: " << result << std::endl;
            return false;
        }

        // 检查输出文件是否生成
        if (!fs::exists(trt_file)) {
            std::cerr << "TensorRT引擎文件生成失败: " << trt_file << std::endl;
            return false;
        }

        std::cout << "TensorRT转换成功: " << trt_file << std::endl;
        return true;
    }
};

// 修复任务结果保存和检索问题
class InceptionDetectionService {
private:
    InceptionConfig config_;
    std::atomic<bool> running_;
    std::atomic<int> task_counter_;

    // 高性能TensorRT推理器池 - 每个工作线程一个推理器
    std::vector<std::unique_ptr<YOLO12TRTInfer>> detector_pool_;
    std::atomic<int> next_detector_index_;

    // 任务队列
    std::queue<DetectionTask> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // 添加任务状态跟踪
    enum class TaskStatus {
        PENDING,    // 等待处理
        PROCESSING, // 正在处理
        COMPLETED,  // 已完成
        FAILED      // 失败
    };

    struct EnhancedTaskResult {
        std::string result;
        std::chrono::steady_clock::time_point submit_time;
        std::chrono::steady_clock::time_point complete_time;
        TaskStatus status;
        std::string error_message;
    };

    // 增强的结果缓存
    std::unordered_map<std::string, EnhancedTaskResult> enhanced_task_results_;
    std::mutex enhanced_results_mutex_;

    // 线程管理
    std::vector<std::thread> worker_threads_;
    std::thread cleanup_thread_;
    std::thread server_thread_;

    // 预处理缓存
    struct ImageCache {
        cv::Mat processed_image;
        std::vector<float> input_tensor;
        std::chrono::steady_clock::time_point cache_time;
    };
    std::unordered_map<std::string, ImageCache> image_cache_;
    std::mutex cache_mutex_;

    // GPU预热
    void warmup_gpu() {
        std::cout << "开始GPU预热..." << std::endl;

        // 创建测试图像
        cv::Mat test_image = cv::Mat::zeros(512, 512, CV_8UC3);
        cv::randu(test_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        std::string temp_path = "temp_warmup.jpg";
        cv::imwrite(temp_path, test_image);

        // 对每个检测器进行预热
        for (size_t i = 0; i < detector_pool_.size(); ++i) {
            try {
                auto start = std::chrono::steady_clock::now();
                detector_pool_[i]->predict(temp_path, false, false, false, false);
                auto end = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "检测器 " << i << " 预热完成，耗时: " << duration.count() << "ms" << std::endl;
            }
            catch (const std::exception& e) {
                std::cerr << "检测器 " << i << " 预热失败: " << e.what() << std::endl;
            }
        }

        // 清理临时文件
        std::remove(temp_path.c_str());

        std::cout << "GPU预热完成！" << std::endl;
    }

    // 修复后的任务添加函数
    std::string add_detection_task_fixed(const std::string& image_path) {
        std::string task_id = generate_task_id();

        // 立即在缓存中创建条目，防止404
        {
            std::lock_guard<std::mutex> lock(enhanced_results_mutex_);
            EnhancedTaskResult task_result;
            task_result.status = TaskStatus::PENDING;
            task_result.submit_time = std::chrono::steady_clock::now();
            task_result.result = "";
            task_result.error_message = "";
            enhanced_task_results_[task_id] = task_result;
        }

        DetectionTask task;
        task.image_path = image_path;
        task.task_id = task_id;
        task.submit_time = std::chrono::steady_clock::now();

        // 添加任务到队列
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            task_queue_.push(std::move(task));
        }

        // 通知工作线程
        queue_cv_.notify_one();

        std::cout << "✓ 任务 " << task_id << " 已添加到队列，图片: " << image_path << std::endl;

        return task_id;
    }

    // 修复后的结果保存函数
    void save_task_result_fixed(const std::string& task_id, const std::string& result, 
                               TaskStatus status, const std::string& error_msg = "") {
        try {
            std::lock_guard<std::mutex> lock(enhanced_results_mutex_);
            
            auto it = enhanced_task_results_.find(task_id);
            if (it != enhanced_task_results_.end()) {
                it->second.result = result;
                it->second.status = status;
                it->second.complete_time = std::chrono::steady_clock::now();
                it->second.error_message = error_msg;
                
                std::cout << "✓ 任务结果已保存: " << task_id << " [状态: " 
                          << static_cast<int>(status) << "]" << std::endl;
            } else {
                std::cerr << "✗ 任务不存在于缓存中: " << task_id << std::endl;
            }
            
            task_counter_++;
        }
        catch (const std::exception& e) {
            std::cerr << "✗ 保存任务结果失败 [" << task_id << "]: " << e.what() << std::endl;
        }
    }

    // 优化的工作线程主循环
    void enhanced_worker_thread(int worker_id) {
        auto& detector = detector_pool_[worker_id];
        std::cout << "🔧 增强工作线程 " << worker_id << " 启动" << std::endl;

        while (running_) {
            DetectionTask task;

            // 获取任务
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { return !task_queue_.empty() || !running_; });

                if (!running_) break;
                if (task_queue_.empty()) continue;

                task = std::move(task_queue_.front());
                task_queue_.pop();
            }

            // 更新任务状态为处理中
            {
                std::lock_guard<std::mutex> lock(enhanced_results_mutex_);
                auto it = enhanced_task_results_.find(task.task_id);
                if (it != enhanced_task_results_.end()) {
                    it->second.status = TaskStatus::PROCESSING;
                }
            }

            std::cout << "🔄 开始处理任务: " << task.task_id << " [线程" << worker_id << "]" << std::endl;

            // 执行检测
            std::string result;
            TaskStatus final_status = TaskStatus::FAILED;
            std::string error_message;

            try {
                auto start_time = std::chrono::steady_clock::now();

                // 文件存在性检查
                if (!UnicodePathUtils::safe_file_exists(task.image_path)) {
                    throw std::runtime_error("Image file not found: " + task.image_path);
                }

                // 执行检测
                result = high_performance_detection_fixed(task.image_path, *detector, worker_id);
                final_status = TaskStatus::COMPLETED;

                auto end_time = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

                std::cout << "✅ 任务完成: " << task.task_id << " [线程" << worker_id 
                          << "] 耗时: " << duration.count() << "ms" << std::endl;
            }
            catch (const std::exception& e) {
                error_message = e.what();
                final_status = TaskStatus::FAILED;

                // 创建错误结果JSON
                nlohmann::json error_result;
                error_result["error"] = "Detection failed";
                error_result["details"] = error_message;
                error_result["task_id"] = task.task_id;
                error_result["worker_id"] = worker_id;
                result = error_result.dump();

                std::cerr << "❌ 任务失败: " << task.task_id << " [线程" << worker_id 
                          << "] 错误: " << error_message << std::endl;
            }

            // 保存结果
            save_task_result_fixed(task.task_id, result, final_status, error_message);

            // 设置promise（如果需要）
            try {
                task.result_promise.set_value(result);
            }
            catch (const std::exception& e) {
                std::cerr << "✗ 设置promise失败 [" << task.task_id << "]: " << e.what() << std::endl;
            }
        }

        std::cout << "🛑 增强工作线程 " << worker_id << " 退出" << std::endl;
    }

    // 修复后的检测函数
    std::string high_performance_detection_fixed(const std::string& image_path, 
                                                YOLO12TRTInfer& detector, int worker_id) {
        try {
            auto overall_start = std::chrono::steady_clock::now();
            
            // 执行检测
            std::string detection_result = detector.predict(image_path, false, false, false, false);
            
            auto detection_end = std::chrono::steady_clock::now();
            auto detection_duration = std::chrono::duration_cast<std::chrono::milliseconds>(detection_end - overall_start);

            // 构建标准化结果
            nlohmann::json result;
            result["classification"] = "DK";
            result["image_path"] = image_path;
            result["worker_id"] = worker_id;
            result["detection_time_ms"] = detection_duration.count();
            result["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            // 解析检测结果
            if (!detection_result.empty()) {
                try {
                    nlohmann::json detection_json = nlohmann::json::parse(detection_result);
                    result["detection"] = detection_json;
                    result["has_detection"] = true;
                }
                catch (const std::exception&) {
                    result["detection"] = detection_result;
                    result["has_detection"] = true;
                    result["parse_warning"] = "Raw string format";
                }
            } else {
                result["detection"] = nlohmann::json::array();
                result["has_detection"] = false;
            }

            return result.dump();
        }
        catch (const std::exception& e) {
            nlohmann::json error_result;
            error_result["error"] = "Detection processing failed";
            error_result["details"] = e.what();
            error_result["image_path"] = image_path;
            error_result["worker_id"] = worker_id;
            return error_result.dump();
        }
    }

    void cleanup_thread() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::minutes(5));  // 每5分钟清理一次

            // 清理结果缓存
            {
                std::lock_guard<std::mutex> lock(enhanced_results_mutex_);
                auto now = std::chrono::steady_clock::now();

                // 清理30分钟前的结果
                auto it = enhanced_task_results_.begin();
                while (it != enhanced_task_results_.end()) {
                    auto age = std::chrono::duration_cast<std::chrono::minutes>(now - it->second.complete_time);
                    if (age.count() > 30) {
                        it = enhanced_task_results_.erase(it);
                    }
                    else {
                        ++it;
                    }
                }
            }

            // 清理图像缓存
            cleanup_cache();
        }
    }

    void cleanup_cache() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto now = std::chrono::steady_clock::now();

        auto it = image_cache_.begin();
        while (it != image_cache_.end()) {
            auto age = std::chrono::duration_cast<std::chrono::minutes>(now - it->second.cache_time);
            if (age.count() > 10) {  // 清理10分钟前的缓存
                it = image_cache_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::string generate_task_id() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
#ifdef _WIN32
        struct tm time_info;
        localtime_s(&time_info, &time_t);
        ss << "task_" << std::put_time(&time_info, "%Y%m%d_%H%M%S")
            << "_" << std::setfill('0') << std::setw(3) << ms.count()
            << "_" << task_counter_.load();
#else
        ss << "task_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
            << "_" << std::setfill('0') << std::setw(3) << ms.count()
            << "_" << task_counter_.load();
#endif

        return ss.str();
    }

    std::string ensureModelFile(const std::string& trt_path, const std::string& trtexec_path) {
        // 首先检查TRT文件是否存在
        if (fs::exists(trt_path)) {
            std::cout << "找到TensorRT引擎文件: " << trt_path << std::endl;
            return trt_path;
        }

        // TRT文件不存在，检查同名ONNX文件
        fs::path trt_file_path(trt_path);
        fs::path onnx_file_path = trt_file_path;
        onnx_file_path.replace_extension(".onnx");

        if (fs::exists(onnx_file_path)) {
            std::cout << "找到ONNX文件，准备转换: " << onnx_file_path << std::endl;

            // 确保目标目录存在
            fs::create_directories(trt_file_path.parent_path());

            // 执行转换
            if (ModelConverter::convertOnnxToTensorRT(onnx_file_path.string(), trt_path, trtexec_path)) {
                return trt_path;
            }
            else {
                throw std::runtime_error("ONNX到TensorRT转换失败: " + onnx_file_path.string());
            }
        }

        // 两种文件都不存在
        throw std::runtime_error("模型文件不存在: " + trt_path + " 和 " + onnx_file_path.string());
    }

    std::string load_html_file(const std::string& filename) {
        fs::path html_path = fs::path(config_.web_root) / filename;

        if (!UnicodePathUtils::safe_file_exists(html_path.string())) {
            std::cerr << "HTML文件不存在: " << html_path << std::endl;
            return create_default_html();
        }

        try {
            std::ifstream file(html_path);
            if (!file.is_open()) {
                std::cerr << "无法打开HTML文件: " << html_path << std::endl;
                return create_default_html();
            }

            std::ostringstream content;
            content << file.rdbuf();
            return content.str();
        }
        catch (const std::exception& e) {
            std::cerr << "读取HTML文件失败: " << e.what() << std::endl;
            return create_default_html();
        }
    }

    std::string create_default_html() {
        return R"(<!DOCTYPE html>
<html>
<head>
    <title>Inception Detection Service</title>
</head>
<body>
    <h1>Inception Image Detection Service</h1>
    <p>服务正在运行，但无法加载完整的Web界面。</p>
    <p>请确保 web/index.html 文件存在。</p>
    <h2>API接口</h2>
    <ul>
        <li>POST /detect - 提交检测任务</li>
        <li>GET /result/{task_id} - 查询结果</li>
        <li>GET /status - 查询服务状态</li>
    </ul>
</body>
</html>)";
    }

    void run_server() {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "WSAStartup失败" << std::endl;
            return;
        }

        SOCKET server_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (server_socket == INVALID_SOCKET) {
            std::cerr << "创建socket失败" << std::endl;
            WSACleanup();
            return;
        }

        // 设置socket选项，允许端口重用
        int opt = 1;
        setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));

        sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(static_cast<u_short>(config_.port));

        if (bind(server_socket, (sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
            std::cerr << "绑定端口失败: " << config_.port << std::endl;
            closesocket(server_socket);
            WSACleanup();
            return;
        }

        if (listen(server_socket, SOMAXCONN) == SOCKET_ERROR) {
            std::cerr << "监听失败" << std::endl;
            closesocket(server_socket);
            WSACleanup();
            return;
        }

        std::cout << "HTTP服务器正在监听端口: " << config_.port << std::endl;

        while (running_) {
            sockaddr_in client_addr;
            int client_addr_len = sizeof(client_addr);

            // 设置非阻塞模式，以便能够响应running_状态变化
            fd_set read_fds;
            FD_ZERO(&read_fds);
            FD_SET(server_socket, &read_fds);

            struct timeval timeout;
            timeout.tv_sec = 1;  // 1秒超时
            timeout.tv_usec = 0;

            int activity = select(0, &read_fds, NULL, NULL, &timeout);
            if (activity <= 0) {
                continue;  // 超时或错误，继续循环检查running_状态
            }

            SOCKET client_socket = accept(server_socket, (sockaddr*)&client_addr, &client_addr_len);
            if (client_socket == INVALID_SOCKET) {
                if (running_) {
                    std::cerr << "接受连接失败" << std::endl;
                }
                continue;
            }

            // 处理客户端请求
            std::thread client_thread(&InceptionDetectionService::handle_client_fixed, this, client_socket);
            client_thread.detach();
        }

        closesocket(server_socket);
        WSACleanup();
#endif
    }

    // 修复后的结果查询处理
    std::string handle_result_request_fixed(const std::string& request) {
        try {
            // 提取task_id
            size_t start = request.find("/result/") + 8;
            size_t end = request.find(" ", start);
            if (end == std::string::npos) end = request.find("\r", start);

            if (start >= request.length() || end <= start) {
                return create_http_response(400, "Bad Request", "application/json",
                    R"({"error": "Invalid task ID format"})");
            }

            std::string task_id = request.substr(start, end - start);
            std::cout << "🔍 查询任务结果: " << task_id << std::endl;

            // 查找任务结果
            std::lock_guard<std::mutex> lock(enhanced_results_mutex_);
            auto it = enhanced_task_results_.find(task_id);

            if (it == enhanced_task_results_.end()) {
                std::cout << "❌ 任务不存在: " << task_id << std::endl;
                nlohmann::json response;
                response["error"] = "Task not found";
                response["task_id"] = task_id;
                return create_http_response(404, "Not Found", "application/json", response.dump());
            }

            const auto& task_result = it->second;

            // 根据任务状态返回响应
            switch (task_result.status) {
                case TaskStatus::PENDING:
                case TaskStatus::PROCESSING: {
                    nlohmann::json response;
                    response["status"] = "processing";
                    response["task_id"] = task_id;
                    response["message"] = "Task is being processed";
                    return create_http_response(202, "Accepted", "application/json", response.dump());
                }

                case TaskStatus::COMPLETED: {
                    nlohmann::json response;
                    response["status"] = "completed";
                    response["task_id"] = task_id;
                    
                    // 验证结果是否为有效JSON
                    try {
                        nlohmann::json detection_result = nlohmann::json::parse(task_result.result);
                        response["result"] = detection_result;
                    }
                    catch (const std::exception&) {
                        response["result"] = task_result.result;
                        response["warning"] = "Result returned as string";
                    }
                    
                    std::cout << "✅ 返回完成结果: " << task_id << std::endl;
                    return create_http_response(200, "OK", "application/json", response.dump());
                }

                case TaskStatus::FAILED: {
                    nlohmann::json response;
                    response["status"] = "failed";
                    response["task_id"] = task_id;
                    response["error"] = task_result.error_message;
                    response["result"] = task_result.result;
                    return create_http_response(200, "OK", "application/json", response.dump());
                }

                default: {
                    nlohmann::json response;
                    response["error"] = "Unknown task status";
                    response["task_id"] = task_id;
                    return create_http_response(500, "Internal Server Error", "application/json", response.dump());
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "❌ 处理结果请求异常: " << e.what() << std::endl;
            nlohmann::json error_response;
            error_response["error"] = "Request processing failed";
            error_response["details"] = e.what();
            return create_http_response(500, "Internal Server Error", "application/json", error_response.dump());
        }
    }

    // 修复后的检测请求处理
    std::string handle_detection_request_fixed(const std::string& request) {
        try {
            // 提取请求体
            size_t body_start = request.find("\r\n\r\n");
            if (body_start == std::string::npos) {
                return create_http_response(400, "Bad Request", "application/json",
                    R"({"error": "Missing request body"})");
            }

            std::string body = request.substr(body_start + 4);

            // 解析JSON请求
            nlohmann::json request_json;
            try {
                request_json = nlohmann::json::parse(body);
            }
            catch (const std::exception& e) {
                std::cerr << "❌ JSON解析错误: " << e.what() << std::endl;
                std::cerr << "请求体内容: " << body << std::endl;
                return create_http_response(400, "Bad Request", "application/json",
                    R"({"error": "Invalid JSON format", "body": ")" + body + R"("})");
            }

            if (!request_json.contains("image_path")) {
                return create_http_response(400, "Bad Request", "application/json",
                    R"({"error": "Missing image_path parameter"})");
            }

            std::string image_path = request_json["image_path"];
            std::cout << "📨 收到检测请求: " << image_path << std::endl;

            // 文件存在性检查
            std::string normalized_path = UnicodePathUtils::normalize_path(image_path);
            bool file_exists = UnicodePathUtils::safe_file_exists(normalized_path);
            
            if (!file_exists) {
                file_exists = UnicodePathUtils::safe_file_exists(image_path);
                if (file_exists) {
                    normalized_path = image_path;
                }
            }

            if (!file_exists) {
                std::cerr << "❌ 文件不存在: " << image_path << std::endl;
                nlohmann::json error_response;
                error_response["error"] = "Image file not found";
                error_response["path"] = image_path;
                return create_http_response(404, "Not Found", "application/json", error_response.dump());
            }

            // 添加检测任务
            std::string task_id = add_detection_task_fixed(normalized_path);

            nlohmann::json response_json;
            response_json["status"] = "accepted";
            response_json["task_id"] = task_id;
            response_json["message"] = "Detection task queued successfully";
            response_json["result_url"] = "/result/" + task_id;

            return create_http_response(202, "Accepted", "application/json", response_json.dump());
        }
        catch (const std::exception& e) {
            std::cerr << "❌ 处理检测请求异常: " << e.what() << std::endl;
            nlohmann::json error_response;
            error_response["error"] = "Request processing error";
            error_response["details"] = e.what();
            return create_http_response(500, "Internal Server Error", "application/json", error_response.dump());
        }
    }

    // 修复后的客户端处理
    void handle_client_fixed(SOCKET client_socket) {
        try {
            char buffer[8192] = { 0 };  // 增大缓冲区
            int bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);

            std::string response;
            if (bytes_received > 0) {
                std::string request(buffer, bytes_received);
                std::cout << "📡 收到请求: " << request.substr(0, std::min(100, (int)request.length())) << "..." << std::endl;

                // 路由请求
                if (request.find("POST /detect") != std::string::npos) {
                    response = handle_detection_request_fixed(request);
                }
                else if (request.find("GET /result/") != std::string::npos) {
                    response = handle_result_request_fixed(request);
                }
                else if (request.find("GET /status") != std::string::npos) {
                    response = handle_status_request();
                }
                else if (request.find("GET /") != std::string::npos) {
                    response = handle_root_request();
                }
                else {
                    response = create_http_response(404, "Not Found", "text/plain", "404 Not Found");
                }
            }
            else {
                response = create_http_response(400, "Bad Request", "text/plain", "400 Bad Request");
            }

            // 发送响应
            int bytes_sent = send(client_socket, response.c_str(), static_cast<int>(response.length()), 0);
            if (bytes_sent == SOCKET_ERROR) {
                std::cerr << "❌ 发送响应失败: " << WSAGetLastError() << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "❌ 客户端处理异常: " << e.what() << std::endl;
            // 发送错误响应
            std::string error_response = create_http_response(500, "Internal Server Error", "text/plain", "Server Error");
            send(client_socket, error_response.c_str(), static_cast<int>(error_response.length()), 0);
        }

        closesocket(client_socket);
    }

    std::string handle_status_request() {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        nlohmann::json status;
        status["service"] = "High-Performance Inception Detection Service (TensorRT)";
        status["status"] = running_ ? "running" : "stopped";
        status["port"] = config_.port;
        status["max_workers"] = config_.max_workers;
        status["detector_pool_size"] = static_cast<int>(detector_pool_.size());
        status["queue_size"] = static_cast<int>(task_queue_.size());
        status["total_tasks_processed"] = task_counter_.load();

        {
            std::lock_guard<std::mutex> results_lock(enhanced_results_mutex_);
            status["cached_results"] = static_cast<int>(enhanced_task_results_.size());
        }

        {
            std::lock_guard<std::mutex> cache_lock(cache_mutex_);
            status["image_cache_size"] = static_cast<int>(image_cache_.size());
        }

        status["classification_engine"] = config_.classification_engine;
        status["detection_engine"] = config_.detection_engine;
        status["performance_mode"] = "high_performance";
        status["gpu_warmed_up"] = true;
        status["lock_free_inference"] = true;

        return create_http_response(200, "OK", "application/json", status.dump());
    }

    std::string handle_root_request() {
        std::string html_content = load_html_file("index.html");
        return create_http_response(200, "OK", "text/html", html_content);
    }

    std::string create_http_response(int status_code, const std::string& status_text,
        const std::string& content_type, const std::string& body) {
        std::ostringstream response;
        response << "HTTP/1.1 " << status_code << " " << status_text << "\r\n";
        response << "Content-Type: " << content_type << "\r\n";
        response << "Content-Length: " << body.length() << "\r\n";
        response << "Access-Control-Allow-Origin: *\r\n";
        response << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
        response << "Access-Control-Allow-Headers: Content-Type\r\n";
        response << "\r\n";
        response << body;
        return response.str();
    }

public:
    InceptionDetectionService(const InceptionConfig& config)
        : config_(config), running_(false), task_counter_(0), next_detector_index_(0)
    {
        try {
            std::cout << "🚀 初始化高性能Inception检测服务..." << std::endl;

            // 验证和准备模型文件
            std::string classification_trt = ensureModelFile(config_.classification_engine, config_.trtexec_path);
            std::string detection_trt = ensureModelFile(config_.detection_engine, config_.trtexec_path);

            std::cout << "📦 创建检测器池，数量: " << config_.max_workers << std::endl;

            // 为每个工作线程创建独立的检测器
            detector_pool_.reserve(config_.max_workers);
            for (int i = 0; i < config_.max_workers; ++i) {
                auto start = std::chrono::steady_clock::now();

                detector_pool_.emplace_back(std::make_unique<YOLO12TRTInfer>(detection_trt));

                auto end = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

                std::cout << "✓ 检测器 " << i << " 初始化完成，耗时: " << duration.count() << "ms" << std::endl;
            }

            std::cout << "模型池初始化成功:" << std::endl;
            std::cout << "- 分类模型: " << classification_trt << std::endl;
            std::cout << "- 检测模型: " << detection_trt << std::endl;
            std::cout << "- 检测器数量: " << detector_pool_.size() << std::endl;
            std::cout << "- Web根目录: " << config_.web_root << std::endl;

            // GPU预热
            warmup_gpu();

        }
        catch (const std::exception& e) {
            std::cerr << "高性能检测服务初始化失败: " << e.what() << std::endl;
            throw;
        }
    }

    ~InceptionDetectionService() {
        stop();
    }

    // 修改启动函数使用修复后的工作线程
    bool start() {
        if (running_) {
            std::cout << "服务已经在运行中" << std::endl;
            return false;
        }

        running_ = true;
        std::cout << "🚀 启动修复后的检测服务..." << std::endl;

        // 启动修复后的工作线程
        for (int i = 0; i < config_.max_workers; ++i) {
            worker_threads_.emplace_back(&InceptionDetectionService::enhanced_worker_thread, this, i);
        }

        // 启动清理线程
        cleanup_thread_ = std::thread([this]() { this->cleanup_thread(); });

        // 启动HTTP服务器线程
        server_thread_ = std::thread([this]() { this->run_server(); });

        std::cout << "⚡ 高性能Inception检测Web服务已启动！" << std::endl;
        std::cout << "🔧 配置信息:" << std::endl;
        std::cout << "   - 服务端口: " << config_.port << std::endl;
        std::cout << "   - 工作线程: " << config_.max_workers << std::endl;
        std::cout << "   - 检测器池: " << detector_pool_.size() << " 个独立实例" << std::endl;
        std::cout << "   - GPU预热: 已完成" << std::endl;
        std::cout << "   - 无锁设计: 已启用" << std::endl;
        std::cout << "📊 性能监控: http://localhost:" << config_.port << "/status" << std::endl;
        std::cout << "🎯 检测接口: http://localhost:" << config_.port << "/detect" << std::endl;

        return true;
    }

    void stop() {
        if (!running_) {
            return;
        }

        std::cout << "正在停止Web服务..." << std::endl;
        running_ = false;

        // 通知所有工作线程退出
        queue_cv_.notify_all();

        // 等待工作线程结束
        for (auto& worker : worker_threads_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        worker_threads_.clear();

        // 等待清理线程结束
        if (cleanup_thread_.joinable()) {
            cleanup_thread_.join();
        }

        // 等待服务器线程结束
        if (server_thread_.joinable()) {
            server_thread_.join();
        }

        std::cout << "Web服务已停止" << std::endl;
    }

    bool is_running() const { return running_; }
    int get_port() const { return config_.port; }
};

void print_usage(const char* program_name) {
    std::cout << "使用方法: " << program_name << " [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  -c, --config <file>     配置文件路径 (默认: C:/DataBase2D/service_config.json)" << std::endl;
    std::cout << "  -h, --help              显示此帮助信息" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << program_name << " --config C:/DataBase2D/service_config.json" << std::endl;
    std::cout << "  " << program_name << std::endl;
}

int main_ai_old(int argc, char* argv[]) {
    try {
        std::string config_file = "C:/DataBase2D/service_config.json";

        // 解析命令行参数
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];

            if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                return 0;
            }
            else if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
                config_file = argv[++i];
            }
        }

        // 加载配置
        InceptionConfig config = InceptionConfig::load_from_file(config_file);

        // 验证配置
        if (config.port <= 0 || config.port > 65535) {
            std::cerr << "错误: 无效的端口号: " << config.port << std::endl;
            return 1;
        }

        if (config.max_workers <= 0 || config.max_workers > 32) {
            std::cerr << "错误: 无效的工作线程数: " << config.max_workers << std::endl;
            return 1;
        }

        std::cout << "=== Inception图像检测Web服务 ===" << std::endl;
        std::cout << "配置文件: " << config_file << std::endl;
        std::cout << "分类引擎: " << config.classification_engine << std::endl;
        std::cout << "检测引擎: " << config.detection_engine << std::endl;
        std::cout << "服务端口: " << config.port << std::endl;
        std::cout << "工作线程: " << config.max_workers << std::endl;
        std::cout << "===============================" << std::endl;

        // 创建并启动服务
        InceptionDetectionService service(config);

        if (!service.start()) {
            std::cerr << "服务启动失败" << std::endl;
            return 1;
        }

        std::cout << "\n服务已启动，按 Ctrl+C 停止服务..." << std::endl;

        // 等待用户中断
        while (service.is_running()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "2D算法模型参数异常，请联系开发人员: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "程序发生未知异常" << std::endl;
        return 1;
    }
}