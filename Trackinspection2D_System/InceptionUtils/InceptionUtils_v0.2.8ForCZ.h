#pragma once

#ifdef INCEPTIONUTILS_EXPORTS
#define INCEPTIONUTILS_API __declspec(dllexport)
#else
#define INCEPTIONUTILS_API __declspec(dllimport)
#endif

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace InceptionUtils {
    INCEPTIONUTILS_API std::string extract_image_num(const std::string& filename);
    INCEPTIONUTILS_API bool is_over_file_exist(const std::string& folder);
    INCEPTIONUTILS_API void mark_folder_over(const std::string& folder);
    INCEPTIONUTILS_API cv::Mat imread_unicode(const std::string& path, int flags = cv::IMREAD_COLOR);
    INCEPTIONUTILS_API bool imwrite_unicode(const std::string& path, const cv::Mat& img, const std::vector<int>& params = {});
    // 用于输出中文消息的辅助函数
    INCEPTIONUTILS_API void print_message(const std::wstring& wideMsg);
    // 新增：将std::string转换为std::wstring并打印
    INCEPTIONUTILS_API void print_message(const std::string& msg);
}