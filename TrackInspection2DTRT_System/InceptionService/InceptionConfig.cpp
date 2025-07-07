#include "InceptionConfig.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

InceptionConfig InceptionConfig::load_from_file(const std::string& config_file) {
    InceptionConfig config;

    // 默认配置
    config.DataBasePath = "C:/DataBase2D";
    config.DataSetPath = "D:/2DImage";
    config.WeightPath = "C:/DataBase2D/weights";
    config.classification_engine = "C1.trt";
    config.detection_engine = "D1.trt";
    config.mark_over_or_not = "true";
    config.CROP_WIDE = 860;
    config.CROP_THRESHOLD = 100;
    config.CENTER_LIMIT = "true";
    config.LIMIT_AREA = 450;
    config.test_image_dir = "";

    config.max_Inspetion_workers = 10;
    config.max_workers = 8;
    config.max_test_images = 50;
    config.stretch_ratio = 2;
    config.confidence_thresh = 0.6f;
    config.iou_thresh = 0.45f;

    if (!fs::exists(config_file)) {
        std::cout << "配置文件不存在，使用默认配置: " << config_file << std::endl;
        return config;
    }

    try {
        std::ifstream file(config_file);
        if (!file.is_open()) {
            std::cerr << "无法打开配置文件: " << config_file << std::endl;
            return config;
        }

        nlohmann::json json_config;
        file >> json_config;

        std::cout << "[调试] 读取到的 JSON 配置:\n" << json_config.dump(4) << std::endl;

        // 读取基础路径配置
        config.DataBasePath = json_config.value("DataBasePath", config.DataBasePath);
        config.DataSetPath = json_config.value("DataSetPath", config.DataSetPath);
        config.WeightPath = json_config.value("WeightPath", config.WeightPath);

        // 读取引擎文件名并拼接完整路径
        std::string classification_filename = json_config.value("classification_engine", "C1.trt");
        std::string detection_filename = json_config.value("detection_engine", "D1.trt");

        // 拼接完整引擎路径
        fs::path weight_dir(config.WeightPath);
        config.classification_engine = (weight_dir / classification_filename).string();
        config.detection_engine = (weight_dir / detection_filename).string();

        // 设置测试图像目录为数据集路径
        config.test_image_dir = config.DataSetPath;

        // 读取其他字符串配置
        config.mark_over_or_not = json_config.value("mark_over_or_not", config.mark_over_or_not);
        config.CENTER_LIMIT = json_config.value("CENTER_LIMIT", config.CENTER_LIMIT);

        // 读取整型配置
        if (json_config.contains("CROP_WIDE") && json_config["CROP_WIDE"].is_number_integer()) {
            config.CROP_WIDE = json_config["CROP_WIDE"].get<int>();
        }

        if (json_config.contains("CROP_THRESHOLD") && json_config["CROP_THRESHOLD"].is_number_integer()) {
            config.CROP_THRESHOLD = json_config["CROP_THRESHOLD"].get<int>();
        }

        if (json_config.contains("LIMIT_AREA") && json_config["LIMIT_AREA"].is_number_integer()) {
            config.LIMIT_AREA = json_config["LIMIT_AREA"].get<int>();
        }

        // 安全读取工作线程数配置
        if (json_config.contains("max_Inspetion_workers") && json_config["max_Inspetion_workers"].is_number_integer()) {
            int workers = json_config["max_Inspetion_workers"].get<int>();
            if (workers > 0 && workers <= 64) {
                config.max_Inspetion_workers = workers;
            }
            else {
                std::cerr << "警告: max_Inspetion_workers 值无效 (" << workers
                    << ")，使用默认值 " << config.max_Inspetion_workers << std::endl;
            }
        }

        if (json_config.contains("max_workers") && json_config["max_workers"].is_number_integer()) {
            int workers = json_config["max_workers"].get<int>();
            if (workers > 0 && workers <= 64) {
                config.max_Inspetion_workers = workers;
            }
            else {
                std::cerr << "警告: max_workers 值无效 (" << workers
                    << ")，使用默认值 " << config.max_Inspetion_workers << std::endl;
            }
        }

        if (json_config.contains("max_test_images") && json_config["max_test_images"].is_number_integer()) {
            int images = json_config["max_test_images"].get<int>();
            if (images > 0 && images <= 10000) {
                config.max_test_images = images;
            }
            else {
                std::cerr << "警告: max_test_images 值无效，使用默认值" << std::endl;
            }
        }

        if (json_config.contains("stretch_ratio") && json_config["stretch_ratio"].is_number()) {
            int ratio = json_config["stretch_ratio"].get<int>();
            if (ratio > 0 && ratio <= 10) {
                config.stretch_ratio = ratio;
            }
            else {
                std::cerr << "警告: stretch_ratio 值无效，使用默认值" << std::endl;
            }
        }

        if (json_config.contains("confidence_thresh") && json_config["confidence_thresh"].is_number()) {
            float thresh = json_config["confidence_thresh"].get<float>();
            if (thresh > 0.0f && thresh <= 1.0f) {
                config.confidence_thresh = thresh;
            }
            else {
                std::cerr << "警告: confidence_thresh 值无效，使用默认值" << std::endl;
            }
        }

        if (json_config.contains("iou_thresh") && json_config["iou_thresh"].is_number()) {
            float thresh = json_config["iou_thresh"].get<float>();
            if (thresh > 0.0f && thresh <= 1.0f) {
                config.iou_thresh = thresh;
            }
            else {
                std::cerr << "警告: iou_thresh 值无效，使用默认值" << std::endl;
            }
        }
        //std::cout << "配置文件加载成功: " << config_file << std::endl;
        //std::cout << ">> 数据库路径: " << config.DataBasePath << std::endl;
        //std::cout << ">> 数据集路径: " << config.DataSetPath << std::endl;
        //std::cout << ">> 权重路径: " << config.WeightPath << std::endl;
        //std::cout << ">> 分类引擎: " << config.classification_engine << std::endl;
        //std::cout << ">> 检测引擎: " << config.detection_engine << std::endl;
        //std::cout << ">> 检测工作线程: " << config.max_Inspetion_workers << std::endl;
        //std::cout << ">> 拉伸比例: " << config.stretch_ratio << std::endl;
        //std::cout << ">> 置信度阈值: " << config.confidence_thresh << std::endl;
        //std::cout << ">> IOU阈值: " << config.iou_thresh << std::endl;

        // 验证关键路径
        if (!fs::exists(config.DataBasePath)) {
            std::cerr << "警告: 数据库目录不存在: " << std::endl;
        }
        if (!fs::exists(config.DataSetPath)) {
            std::cerr << "警告: 数据集目录不存在: "  << std::endl;
        }
        if (!fs::exists(config.classification_engine)) {
            std::cerr << "警告: 分类引擎文件不存在: "  << std::endl;
        }
        if (!fs::exists(config.detection_engine)) {
            std::cerr << "警告: 检测引擎文件不存在: "  << std::endl;
        }





    }
    catch (const nlohmann::json::exception& e) {
        std::cerr << "JSON 解析错误: " << e.what() << std::endl;
        std::cout << "使用默认配置" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "配置文件解析失败: " << e.what() << std::endl;
        std::cout << "使用默认配置" << std::endl;
    }

    return config;
}

// 验证配置的有效性
bool InceptionConfig::validate() const {
    bool valid = true;

    if (max_Inspetion_workers <= 0 || max_Inspetion_workers > 64) {
        std::cerr << "配置验证失败: max_Inspetion_workers 无效 (" << max_Inspetion_workers << ")" << std::endl;
        valid = false;
    }

    if (max_test_images <= 0 || max_test_images > 10000) {
        std::cerr << "配置验证失败: max_test_images 无效 (" << max_test_images << ")" << std::endl;
        valid = false;
    }

    if (stretch_ratio <= 0 || stretch_ratio > 10) {
        std::cerr << "配置验证失败: stretch_ratio 无效 (" << stretch_ratio << ")" << std::endl;
        valid = false;
    }

    if (confidence_thresh <= 0.0f || confidence_thresh > 1.0f) {
        std::cerr << "配置验证失败: confidence_thresh 无效 (" << confidence_thresh << ")" << std::endl;
        valid = false;
    }

    if (iou_thresh <= 0.0f || iou_thresh > 1.0f) {
        std::cerr << "配置验证失败: iou_thresh 无效 (" << iou_thresh << ")" << std::endl;
        valid = false;
    }

    if (CROP_WIDE <= 0 || CROP_WIDE > 2000) {
        std::cerr << "配置验证失败: CROP_WIDE 无效 (" << CROP_WIDE << ")" << std::endl;
        valid = false;
    }

    if (CROP_THRESHOLD <= 0 || CROP_THRESHOLD > 255) {
        std::cerr << "配置验证失败: CROP_THRESHOLD 无效 (" << CROP_THRESHOLD << ")" << std::endl;
        valid = false;
    }

    if (LIMIT_AREA <= 0 || LIMIT_AREA > 1000) {
        std::cerr << "配置验证失败: LIMIT_AREA 无效 (" << LIMIT_AREA << ")" << std::endl;
        valid = false;
    }

    return valid;
}