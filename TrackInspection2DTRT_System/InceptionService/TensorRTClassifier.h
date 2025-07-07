#pragma once
#pragma once

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <future>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>


// 单线程 TensorRT 分类器
class TensorRTClassifier {
public:
    TensorRTClassifier(const std::string& engine_path, 
                      cv::Size input_size = cv::Size(224, 224));
    ~TensorRTClassifier();

    std::string classify(const std::string& image_path);
    std::pair<std::string, float> classify_with_confidence(const std::string& image_path);

private:
    bool loadEngine(const std::string& engine_path);
    std::vector<float> preprocessImage(const cv::Mat& img);
    std::string postprocess(const std::vector<float>& output);

    // TensorRT 组件
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // CUDA 资源
    void* input_buffer_;
    void* output_buffer_;
    cudaStream_t stream_;

    // 模型参数
    cv::Size input_size_;
    size_t input_bytes_;
    size_t output_bytes_;
    std::string input_name_;
    std::string output_name_;
    
    // 分类标签
    std::vector<std::string> class_names_;
};

// 多线程分类器管理器
class MultiThreadClassifier {
public:
    MultiThreadClassifier(const std::string& engine_path, 
                         int num_threads = 4,
                         cv::Size input_size = cv::Size(224, 224));
    ~MultiThreadClassifier();

    // 同步分类
    std::string classify(const std::string& image_path);
    std::vector<std::string> classify_batch(const std::vector<std::string>& image_paths);

    // 异步分类
    std::future<std::string> classify_async(const std::string& image_path);
    std::future<std::vector<std::string>> classify_batch_async(const std::vector<std::string>& image_paths);

    // 状态查询
    size_t get_queue_size() const;
    int get_active_threads() const;

private:
    struct Task {
        std::string image_path;
        std::promise<std::string> promise;

        Task() = default;
        Task(Task&& other) noexcept = default;
        Task& operator=(Task&& other) noexcept = default;

        // 禁用拷贝构造和拷贝赋值
        Task(const Task&) = delete;
        Task& operator=(const Task&) = delete;
    };

    void worker_thread(int thread_id);

    std::vector<std::unique_ptr<TensorRTClassifier>> classifiers_;
    std::vector<std::thread> workers_;
    
    std::queue<Task> task_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    
    std::atomic<bool> running_;
    std::string engine_path_;
    cv::Size input_size_;
    int num_threads_;
};