#include "TensorRTClassifier.h"
#include <fstream>
#include <iostream>
#include <algorithm>

// 分类标签定义（根据您的模型调整）
const std::vector<std::string> CLASS_LABELS = {
    "YC", "DK", "BM", "HF", "CS", "ZC", "GF", "GD"
};

TensorRTClassifier::TensorRTClassifier(const std::string& engine_path, cv::Size input_size)
    : input_size_(input_size)
    , input_buffer_(nullptr)
    , output_buffer_(nullptr)
    , stream_(nullptr)
    , class_names_(CLASS_LABELS)
{
    // 创建 CUDA 流
    cudaStreamCreate(&stream_);
    
    // 加载 TensorRT 引擎
    if (!loadEngine(engine_path)) {
        throw std::runtime_error("Failed to load TensorRT engine: " + engine_path);
    }
    
    std::cout << "TensorRT 分类器初始化成功，输入尺寸: " << input_size_.width << "x" << input_size_.height << std::endl;
}

TensorRTClassifier::~TensorRTClassifier() {
    if (input_buffer_) cudaFree(input_buffer_);
    if (output_buffer_) cudaFree(output_buffer_);
    if (stream_) cudaStreamDestroy(stream_);
}

bool TensorRTClassifier::loadEngine(const std::string& engine_path) {
    // 读取引擎文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "无法读取引擎文件: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();


    // 反序列化引擎
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
        std::cerr << "反序列化 TensorRT 引擎失败" << std::endl;
        return false;
    }

    // 创建执行上下文
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "创建 TensorRT 执行上下文失败" << std::endl;
        return false;
    }

    // 获取输入输出信息（TensorRT 10 新 API）
    int32_t nb_io_tensors = engine_->getNbIOTensors();
    for (int32_t i = 0; i < nb_io_tensors; ++i) {
        const char* tensor_name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);

        if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = tensor_name;
        } else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
            output_name_ = tensor_name;
        }
    }

    // 计算缓冲区大小
    input_bytes_ = 1 * 3 * input_size_.height * input_size_.width * sizeof(float);
    
    nvinfer1::Dims output_dims = engine_->getTensorShape(output_name_.c_str());
    size_t output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size *= output_dims.d[i];
    }
    output_bytes_ = output_size * sizeof(float);

    // 分配 GPU 内存
    cudaMalloc(&input_buffer_, input_bytes_);
    cudaMalloc(&output_buffer_, output_bytes_);

    std::cout << "TensorRT 引擎加载成功，输入: " << input_name_ << ", 输出: " << output_name_ << std::endl;
    return true;
}

std::vector<float> TensorRTClassifier::preprocessImage(const cv::Mat& img) {
    // 调整图像大小
    cv::Mat resized;
    cv::resize(img, resized, input_size_);

    // BGR 转 RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // 归一化到 [0, 1]
    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32F, 1.0 / 255.0);

    // 转换为 CHW 格式
    std::vector<cv::Mat> channels(3);
    cv::split(normalized, channels);

    std::vector<float> input_tensor;
    input_tensor.reserve(input_size_.height * input_size_.width * 3);

    // 按 RGB 顺序排列
    for (int c = 0; c < 3; ++c) {
        const float* data = channels[c].ptr<float>();
        input_tensor.insert(input_tensor.end(), data, 
                           data + input_size_.height * input_size_.width);
    }

    return input_tensor;
}

std::string TensorRTClassifier::postprocess(const std::vector<float>& output) {
    // 找到最大值的索引
    auto max_it = std::max_element(output.begin(), output.end());
    int class_id = static_cast<int>(std::distance(output.begin(), max_it));
    
    // 返回对应的类别名称
    if (class_id >= 0 && class_id < static_cast<int>(class_names_.size())) {
        return class_names_[class_id];
    }
    
    return "Unknown";
}

std::string TensorRTClassifier::classify(const std::string& image_path) {
    // 读取图像
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像: " + image_path);
    }

    // 预处理
    std::vector<float> input_tensor = preprocessImage(img);

    // 复制数据到 GPU
    cudaMemcpyAsync(input_buffer_, input_tensor.data(), input_bytes_, 
                    cudaMemcpyHostToDevice, stream_);

    // 设置输入形状（TensorRT 10 新 API）
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 1;  // batch size
    input_dims.d[1] = 3;  // channels
    input_dims.d[2] = input_size_.height;
    input_dims.d[3] = input_size_.width;
    context_->setInputShape(input_name_.c_str(), input_dims);

    // 设置张量地址
    context_->setTensorAddress(input_name_.c_str(), input_buffer_);
    context_->setTensorAddress(output_name_.c_str(), output_buffer_);

    // 执行推理
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT 推理失败");
    }

    // 复制输出数据
    std::vector<float> output_data(output_bytes_ / sizeof(float));
    cudaMemcpyAsync(output_data.data(), output_buffer_, output_bytes_, 
                    cudaMemcpyDeviceToHost, stream_);

    // 等待推理完成
    cudaStreamSynchronize(stream_);

    // 后处理
    return postprocess(output_data);
}

std::pair<std::string, float> TensorRTClassifier::classify_with_confidence(const std::string& image_path) {
    // 读取图像
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像: " + image_path);
    }

    // 预处理
    std::vector<float> input_tensor = preprocessImage(img);

    // GPU 推理过程（同上）
    cudaMemcpyAsync(input_buffer_, input_tensor.data(), input_bytes_, 
                    cudaMemcpyHostToDevice, stream_);

    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 1;
    input_dims.d[1] = 3;
    input_dims.d[2] = input_size_.height;
    input_dims.d[3] = input_size_.width;
    context_->setInputShape(input_name_.c_str(), input_dims);

    context_->setTensorAddress(input_name_.c_str(), input_buffer_);
    context_->setTensorAddress(output_name_.c_str(), output_buffer_);

    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT 推理失败");
    }

    std::vector<float> output_data(output_bytes_ / sizeof(float));
    cudaMemcpyAsync(output_data.data(), output_buffer_, output_bytes_, 
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // 后处理 - 返回类别和置信度
    auto max_it = std::max_element(output_data.begin(), output_data.end());
    int class_id = static_cast<int>(std::distance(output_data.begin(), max_it));
    float confidence = *max_it;
    
    std::string class_name = (class_id >= 0 && class_id < static_cast<int>(class_names_.size())) 
                            ? class_names_[class_id] : "Unknown";
    
    return {class_name, confidence};
}

// 多线程分类器实现
MultiThreadClassifier::MultiThreadClassifier(const std::string& engine_path, 
                                           int num_threads, cv::Size input_size)
    : engine_path_(engine_path)
    , input_size_(input_size)
    , num_threads_(num_threads)
    , running_(true)
{
    std::cout << "初始化多线程分类器，线程数: " << num_threads_ << std::endl;

    // 为每个线程创建独立的分类器
    classifiers_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; ++i) {
        try {
            auto classifier = std::make_unique<TensorRTClassifier>(engine_path_, input_size_);
            classifiers_.emplace_back(std::move(classifier));
            std::cout << "分类器线程 " << i << " 初始化完成" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "分类器线程 " << i << " 初始化失败: " << e.what() << std::endl;
            throw;
        }
    }

    // 启动工作线程
    workers_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; ++i) {
        workers_.emplace_back(&MultiThreadClassifier::worker_thread, this, i);
    }

    std::cout << "多线程分类器启动完成！" << std::endl;
}

MultiThreadClassifier::~MultiThreadClassifier() {
    running_ = false;
    condition_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    std::cout << "多线程分类器已停止" << std::endl;
}

void MultiThreadClassifier::worker_thread(int thread_id) {
    auto& classifier = classifiers_[thread_id];
    std::cout << "工作线程 " << thread_id << " 启动" << std::endl;

    while (running_) {
        Task task;

        // 获取任务
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return !task_queue_.empty() || !running_; });

            if (!running_) break;
            if (task_queue_.empty()) continue;

            task = std::move(task_queue_.front());
            task_queue_.pop();
        }

        // 执行分类
        try {
            std::string result = classifier->classify(task.image_path);
            task.promise.set_value(result);
        } catch (const std::exception& e) {
            std::cerr << "线程 " << thread_id << " 分类失败: " << e.what() << std::endl;
            task.promise.set_value("Error");
        }
    }

    std::cout << "工作线程 " << thread_id << " 退出" << std::endl;
}

std::string MultiThreadClassifier::classify(const std::string& image_path) {
    auto future = classify_async(image_path);
    return future.get();
}

std::future<std::string> MultiThreadClassifier::classify_async(const std::string& image_path) {
    Task task;
    task.image_path = image_path;
    auto future = task.promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
    }

    condition_.notify_one();
    return future;
}

std::vector<std::string> MultiThreadClassifier::classify_batch(const std::vector<std::string>& image_paths) {
    auto future = classify_batch_async(image_paths);
    return future.get();
}

std::future<std::vector<std::string>> MultiThreadClassifier::classify_batch_async(const std::vector<std::string>& image_paths) {
    auto promise = std::make_shared<std::promise<std::vector<std::string>>>();
    auto future = promise->get_future();
    
    auto results = std::make_shared<std::vector<std::string>>(image_paths.size());
    auto counter = std::make_shared<std::atomic<size_t>>(0);
    
    for (size_t i = 0; i < image_paths.size(); ++i) {
        auto single_future = classify_async(image_paths[i]);
        
        std::thread([single_future = std::move(single_future), results, counter, promise, i, total = image_paths.size()]() mutable {
            try {
                (*results)[i] = single_future.get();
            } catch (...) {
                (*results)[i] = "Error";
            }
            
            if (counter->fetch_add(1) + 1 == total) {
                promise->set_value(*results);
            }
        }).detach();
    }
    
    return future;
}

size_t MultiThreadClassifier::get_queue_size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
}

int MultiThreadClassifier::get_active_threads() const {
    return num_threads_;
}