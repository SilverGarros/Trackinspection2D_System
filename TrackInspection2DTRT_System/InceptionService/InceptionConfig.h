#pragma once
#include <string>

struct InceptionConfig {
    std::string DataBasePath;
    std::string DataSetPath;    
    std::string WeightPath;
    std::string classification_engine;
    std::string detection_engine;
    std::string mark_over_or_not;

    int CROP_WIDE;
    int CROP_THRESHOLD;
    std::string CENTER_LIMIT;
    int LIMIT_AREA;

    std::string test_image_dir;

    int max_Inspetion_workers = 8;
    int max_workers = 8;
    int max_test_images = 50;
    int stretch_ratio = 2;
    float confidence_thresh = 0.6f;
    float iou_thresh = 0.45f;

    // 静态成员函数声明
    static InceptionConfig load_from_file(const std::string& config_file);

    // 验证配置有效性
    bool validate() const;
};