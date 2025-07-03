#include <windows.h>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
#include "FunctionDLL.h"

int main() {
    // 屏蔽OpenCV的INFO日志
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    std::vector<std::pair<std::string, std::string>> img_pairs = {
        {"D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\true\\2.png", "D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\test\\true\\2_old.png"},
        {"D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\001.png", "D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\001_old.png"},
        {"D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\002.png", "D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\002_old.png"},
        {"D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\003.png", "D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\003_old.png"},
        {"D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\005.png", "D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\005_old.png"},
        {"D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\006.png", "D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\006_old.png"},
        {"D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\007.png", "D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\007_old.png"},
        {"D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\008.png", "D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\008_old.png"},
        {"D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\009.png", "D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\009_old.png"},
        {"D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\10.png", "D:\\LuHang_System\\FunctionDLL\\x64\\Debug\\中文测试\\test\\10_old.png"}
    };

    for (size_t i = 0; i < img_pairs.size(); ++i) {
        const auto& img1 = img_pairs[i].first;
        const auto& img2 = img_pairs[i].second;
        // std::cout << "测试文件: " << img1 << " 存在? " << std::filesystem::exists(img1) << std::endl;



        std::cout << "==== 测试组 " << i << img1 << " ====" << std::endl;

        float sim_hog_cos = area_anomaly_calibration(img1, img2, 3, 8, 6, 12, "cosine");
        std::cout << "HOG cosine相似度: " << sim_hog_cos << std::endl;
        float sim_hoglbp_cos = area_anomaly_calibration_based_hoglbp(img1, img2, 3, 8, 6, 12, "cosine");
        std::cout << "HOG+LBP cosine相似度: " << sim_hoglbp_cos << std::endl;

        float sim_hog_euc = area_anomaly_calibration(img1, img2, 3, 8, 6, 12, "euclidean");
        std::cout << "HOG euclidean相似度: " << sim_hog_euc << std::endl;
        float sim_hoglbp_euc = area_anomaly_calibration_based_hoglbp(img1, img2, 3, 8, 6, 12, "euclidean");
        std::cout << "HOG+LBP euclidean相似度: " << sim_hoglbp_euc << std::endl;

        sim_hog_euc = area_anomaly_calibration(img1, img2, 3, 8, 6, 12, "chi2");
        std::cout << "HOG chi2相似度: " << sim_hog_euc << std::endl;
        sim_hoglbp_euc = area_anomaly_calibration_based_hoglbp(img1, img2, 3, 8, 6, 12, "chi2");
        std::cout << "HOG+LBP chi2相似度: " << sim_hoglbp_euc << std::endl;
    }

    return 0;
}