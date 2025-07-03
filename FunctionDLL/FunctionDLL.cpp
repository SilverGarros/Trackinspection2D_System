#include "pch.h"
#include "FunctionDLL.h"
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <optional>
#include <filesystem>
#include <regex>
#include <string>

// 字符串转宽字符串（支持UTF-8）
std::wstring utf8_to_wstring(const std::string& str) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    if (wlen == 0) return L"";
    std::wstring wstr(wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], wlen);
    if (!wstr.empty() && wstr.back() == L'\0') wstr.pop_back();
    return wstr;
}

// 支持中文路径的OpenCV图片读取
cv::Mat imread_unicode(const std::string& imagePath, int flags = cv::IMREAD_COLOR) {
    // 先尝试直接读取（部分OpenCV版本支持UTF-8路径）
    cv::Mat img = cv::imread(imagePath, flags);
    if (!img.empty()) return img;

    // 用宽字符流读取文件内容
    std::wstring wpath = utf8_to_wstring(imagePath);
    FILE* fp = nullptr;
    if (_wfopen_s(&fp, wpath.c_str(), L"rb") != 0 || !fp) return cv::Mat();

    std::vector<uchar> buffer;
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    buffer.resize(len);
    fread(buffer.data(), 1, len, fp);
    fclose(fp);

    return cv::imdecode(buffer, flags);
}

// 检测红色矩形内边缘
cv::Rect get_red_rect_inner_coords(const cv::Mat& img, int margin = 3) {
    cv::Mat hsv, mask1, mask2, mask;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 70, 50), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(170, 70, 50), cv::Scalar(180, 255, 255), mask2);
    mask = mask1 | mask2;
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat::ones(5, 5, CV_8U));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return cv::Rect();
    auto max_it = std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    cv::Rect rect = cv::boundingRect(*max_it);
    rect.x += margin;
    rect.y += margin;
    rect.width = std::max(0, rect.width - 2 * margin);
    rect.height = std::max(0, rect.height - 2 * margin);
    return rect;
}

// 裁剪图片
cv::Mat crop_img_with_rect(const cv::Mat& img, const cv::Rect& rect) {
    if (rect.width <= 0 || rect.height <= 0) return cv::Mat();
    return img(rect).clone();
}

// 计算归一化直方图
std::vector<float> calc_hist(const cv::Mat& lbp, int n_bins) {
    std::vector<float> hist(n_bins, 0.0f);
    for (int y = 0; y < lbp.rows; ++y)
        for (int x = 0; x < lbp.cols; ++x)
            hist[lbp.at<uchar>(y, x)] += 1.0f;
    float total = static_cast<float>(lbp.rows * lbp.cols);
    if (total > 0) {
        for (auto& v : hist) v /= total;
    }
    return hist;
}

// 基于欧氏距离获取相似度
float l2_similarity(const std::vector<float>& h1, const std::vector<float>& h2) {
    float dist = 0.0f;
    for (size_t i = 0; i < h1.size(); ++i)
        dist += (h1[i] - h2[i]) * (h1[i] - h2[i]);
    dist = std::sqrt(dist);
    return 1.0f / (1.0f + dist);
    //// 归一化
    //float norm_dist = dist / std::sqrt(static_cast<float>(h1.size()));
    //// 指数映射，alpha可调，建议0.8~1.2
    //float alpha = 1.0f;
    //return std::exp(-alpha * norm_dist);
}

void l2_normalize(std::vector<float>& desc) {
    float norm = 0.0f;
    for (float v : desc) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (float& v : desc) v /= norm;
    }
}
// 余弦相似度
float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    if (norm1 == 0 || norm2 == 0) return 0.0f;
    return dot / (std::sqrt(norm1) * std::sqrt(norm2));
}

// 卡尔距离相似度
float chi2_similarity(const std::vector<float>& hist1, const std::vector<float>& hist2) {
    float chi2 = 0.0f;
    for (size_t i = 0; i < hist1.size(); ++i) {
        float sum = hist1[i] + hist2[i] + 1e-10f;
        float diff = hist1[i] - hist2[i];
        chi2 += (diff * diff) / sum;
    }
    chi2 *= 0.05f;
    return 1.0f / (1.0f + chi2);

    //float norm_chi2 = chi2 / static_cast<float>(hist1.size());    // 归一化
    //float beta = 0.8f;     // 指数映射，beta可调，建议0.8~1.2
    //return std::exp(-beta * norm_chi2);
}


// HOG特征相似度(cv::Mat重载)
std::optional<float> compute_hog_similarity(const cv::Mat& img1, const cv::Mat& img2,
    int pixels_per_cell = 8, int cells_per_block = 6, int orientations = 12, const std::string& metric = "euclidean") {
    // 检查输入
    if (img1.empty() || img2.empty()) return std::nullopt;

    // 转灰度
    cv::Mat gray1, gray2;
    if (img1.channels() == 3)
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    else
        gray1 = img1.clone();
    if (img2.channels() == 3)
        cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    else
        gray2 = img2.clone();

    // 统一尺寸
    int h = std::min(gray1.rows, gray2.rows);
    int w = std::min(gray1.cols, gray2.cols);
    //gray1 = gray1(cv::Rect(0, 0, w, h));
    //gray2 = gray2(cv::Rect(0, 0, w, h));

    // HOG参数
    cv::Size winSize(w, h);
    cv::Size blockSize(pixels_per_cell * cells_per_block, pixels_per_cell * cells_per_block);
    cv::Size blockStride(pixels_per_cell, pixels_per_cell);
    cv::Size cellSize(pixels_per_cell, pixels_per_cell);

    cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, orientations);

    std::vector<float> desc1, desc2;
    hog.compute(gray1, desc1);
    hog.compute(gray2, desc2);

    // 手动L2归一化
    l2_normalize(desc1);
    l2_normalize(desc2);

    if (metric == "cosine") {
        return cosine_similarity(desc1, desc2);
    }
    else if (metric == "euclidean") {
        return l2_similarity(desc1, desc2);
    }
    else if (metric == "chi2") {
        return chi2_similarity(desc1, desc2);
    }
    else {
        std::cerr << "不支持的metric类型: " << metric << std::endl;
        return std::nullopt;
    }
}
// HOG特征相似度(路径重载)
std::optional<float> compute_hog_similarity(const std::string& img1_path, const std::string& img2_path,
    int pixels_per_cell, int cells_per_block, int orientations, const std::string& metric) {
    cv::Mat img1 = imread_unicode(img1_path, cv::IMREAD_COLOR);
    cv::Mat img2 = imread_unicode(img2_path, cv::IMREAD_COLOR);
    return compute_hog_similarity(img1, img2, pixels_per_cell, cells_per_block, orientations, metric);
}

// 计算LBP特征（8邻域，256维直方图归一化）
cv::Mat compute_lbp(const cv::Mat& gray) {
    cv::Mat lbp = cv::Mat::zeros(gray.rows - 2, gray.cols - 2, CV_8UC1);
    for (int i = 1; i < gray.rows - 1; i++) {
        for (int j = 1; j < gray.cols - 1; j++) {
            uchar center = gray.at<uchar>(i, j);
            unsigned char code = 0;
            code |= (gray.at<uchar>(i - 1, j - 1) > center) << 7;
            code |= (gray.at<uchar>(i - 1, j) > center) << 6;
            code |= (gray.at<uchar>(i - 1, j + 1) > center) << 5;
            code |= (gray.at<uchar>(i, j + 1) > center) << 4;
            code |= (gray.at<uchar>(i + 1, j + 1) > center) << 3;
            code |= (gray.at<uchar>(i + 1, j) > center) << 2;
            code |= (gray.at<uchar>(i + 1, j - 1) > center) << 1;
            code |= (gray.at<uchar>(i, j - 1) > center) << 0;
            lbp.at<uchar>(i - 1, j - 1) = code;
        }
    }
    return lbp;
}

// HOG+LBP特征相似度(cv::Mat重载)
std::optional<float> compute_hoglbp_similarity(const cv::Mat& img1, const cv::Mat& img2,
    int pixels_per_cell = 8, int cells_per_block = 6, int orientations = 12, const std::string& metric = "cosine")
{
    if (img1.empty() || img2.empty()) return std::nullopt;

    // resize到统一尺寸
    cv::Size hogSize(128, 64);
    cv::Mat img1_resized, img2_resized;
    cv::resize(img1, img1_resized, hogSize);
    cv::resize(img2, img2_resized, hogSize);

    // HOG特征
    cv::Mat gray1, gray2;
    if (img1_resized.channels() == 3)
        cv::cvtColor(img1_resized, gray1, cv::COLOR_BGR2GRAY);
    else
        gray1 = img1_resized.clone();
    if (img2_resized.channels() == 3)
        cv::cvtColor(img2_resized, gray2, cv::COLOR_BGR2GRAY);
    else
        gray2 = img2_resized.clone();

    cv::Size winSize = hogSize;
    cv::Size blockSize(pixels_per_cell * cells_per_block, pixels_per_cell * cells_per_block);
    cv::Size blockStride(pixels_per_cell, pixels_per_cell);
    cv::Size cellSize(pixels_per_cell, pixels_per_cell);

    cv::HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, orientations);
    std::vector<float> desc1, desc2;
    hog.compute(gray1, desc1);
    hog.compute(gray2, desc2);

    // LBP特征（256维直方图归一化）
    cv::Mat lbp1 = compute_lbp(gray1);
    cv::Mat lbp2 = compute_lbp(gray2);
    std::vector<float> lbp_hist1 = calc_hist(lbp1, 256);
    std::vector<float> lbp_hist2 = calc_hist(lbp2, 256);

    // 融合特征
    std::vector<float> feat1, feat2;
    feat1.reserve(desc1.size() + lbp_hist1.size());
    feat2.reserve(desc2.size() + lbp_hist2.size());
    feat1.insert(feat1.end(), desc1.begin(), desc1.end());
    feat1.insert(feat1.end(), lbp_hist1.begin(), lbp_hist1.end());
    feat2.insert(feat2.end(), desc2.begin(), desc2.end());
    feat2.insert(feat2.end(), lbp_hist2.begin(), lbp_hist2.end());

    if (feat1.size() != feat2.size()) return std::nullopt;
    // 手动L2归一化
    l2_normalize(feat1);
    l2_normalize(feat2);

    if (metric == "cosine") {
        return cosine_similarity(feat1, feat2);
    }
    else if (metric == "euclidean") {
        return l2_similarity(feat1, feat2);
    }
    else if (metric == "chi2") {
        return chi2_similarity(feat1, feat2);
    }
    else {
        std::cerr << "不支持的metric类型: " << metric << std::endl;
        return std::nullopt;
    }
}

// HOG+LBP特征相似度(路径重载)
std::optional<float> compute_hoglbp_similarity(const std::string& img1_path, const std::string& img2_path,
    int pixels_per_cell, int cells_per_block, int orientations, const std::string& metric)
{
    cv::Mat img1 = imread_unicode(img1_path, cv::IMREAD_COLOR);
    cv::Mat img2 = imread_unicode(img2_path, cv::IMREAD_COLOR);
    return compute_hoglbp_similarity(img1, img2, pixels_per_cell, cells_per_block, orientations, metric);
}

// 基于HOG相似度的异常区域验证
float area_anomaly_calibration(const std::string& defect_Display_Diagram_path, const std::string& defect_Old_Diagram_path, int margin = 3, 
    int pixels_per_cell = 8, int cells_per_block = 6, int orientations = 12, const std::string& metric = "cosine") {
    // 先获取红色矩形区域
    cv::Mat display_img = imread_unicode(defect_Display_Diagram_path, cv::IMREAD_COLOR);
    if (display_img.empty()) {
        std::cout << "未找到缺陷显示图: " << defect_Display_Diagram_path << std::endl;
        return 0.0f;
    }
    cv::Rect coords = get_red_rect_inner_coords(display_img, margin);
    if (coords.width == 0 || coords.height == 0) {
        std::cout << "未找到红色矩形框" << std::endl;
        return 0.0f;
    }

    // 裁剪两张图片
    cv::Mat display_crop = crop_img_with_rect(display_img, coords);
    cv::Mat old_img = imread_unicode(defect_Old_Diagram_path, cv::IMREAD_COLOR);
    if (old_img.empty()) {
        std::cout << "未找到旧图: " << defect_Old_Diagram_path << std::endl;
        return 0.0f;
    }
    cv::Mat old_crop = crop_img_with_rect(old_img, coords);

    // 临时保存裁剪结果到临时文件（compute_hog_similarity 只接受路径）
    std::string tmp1 = "tmp_display_crop.png";
    std::string tmp2 = "tmp_old_crop.png";
    cv::imwrite(tmp1, display_crop);
    cv::imwrite(tmp2, old_crop);

    // 计算HOG相似度
    std::optional<float> defect_hog = compute_hog_similarity(tmp1, tmp2, pixels_per_cell, cells_per_block, orientations, metric);
    if (defect_hog) {
        //std::cout << defect_Display_Diagram_path << "缺陷显示图与旧图" << defect_Old_Diagram_path
        //    << "的HOG相似度(pixels_per_cell = 8, cells_per_block = 6, orientations = 12, metric='euclidean'): "
        //    << *defect_hog << std::endl;
        return *defect_hog;
    }
    else {
        std::cout << "HOG相似度计算失败" << std::endl;
        return 0.0f;
    }

    // 删除临时文件
    //std::remove(tmp1.c_str());
    //std::remove(tmp2.c_str());
}

// 基于HOG+LBP相似度的异常区域验证
float area_anomaly_calibration_based_hoglbp(const std::string& defect_Display_Diagram_path, const std::string& defect_Old_Diagram_path, int margin = 3,
    int pixels_per_cell = 8, int cells_per_block = 6, int orientations = 12, const std::string& metric = "cosine")
{
    // 先获取红色矩形区域
    cv::Mat display_img = imread_unicode(defect_Display_Diagram_path, cv::IMREAD_COLOR);
    if (display_img.empty()) {
        std::cout << "未找到缺陷显示图: " << defect_Display_Diagram_path << std::endl;
        return 0.0f;
    }
    cv::Rect coords = get_red_rect_inner_coords(display_img, margin);
    if (coords.width == 0 || coords.height == 0) {
        std::cout << "未找到红色矩形框" << std::endl;
        return 0.0f;
    }

    // 裁剪两张图片
    cv::Mat display_crop = crop_img_with_rect(display_img, coords);
    cv::Mat old_img = imread_unicode(defect_Old_Diagram_path, cv::IMREAD_COLOR);
    if (old_img.empty()) {
        std::cout << "未找到旧图: " << defect_Old_Diagram_path << std::endl;
        return 0.0f;
    }
    cv::Mat old_crop = crop_img_with_rect(old_img, coords);

    // 临时保存裁剪结果到临时文件（compute_hoglbp_similarity 只接受路径）
    std::string tmp1 = "tmp_display_crop.png";
    std::string tmp2 = "tmp_old_crop.png";
    cv::imwrite(tmp1, display_crop);
    cv::imwrite(tmp2, old_crop);

    // 计算HOG+LBP相似度
    std::optional<float> sim = compute_hoglbp_similarity(tmp1, tmp2, pixels_per_cell, cells_per_block, orientations, metric);
    if (sim) {
        return *sim;
    }
    else {
        std::cout << "HOG+LBP相似度计算失败" << std::endl;
        return 0.0f;
    }
}

// 获取指定路径下的最大序号
int get_next_save_index(const std::string& dir) {
    namespace fs = std::filesystem;
    int max_idx = 0;
    std::regex re(R"((\d+)_.*\.png)");
    if (!fs::exists(dir)) return 1;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::smatch m;
        std::string fname = entry.path().filename().string();
        if (std::regex_match(fname, m, re)) {
            int idx = std::stoi(m[1].str());
            if (idx > max_idx) max_idx = idx;
        }
    }
    return max_idx + 1;
}

// 带自动保存的基于HOG相似度的异常区域验证(使用原图扩大范围)
float area_anomaly_calibration_with_save(const std::string& defect_Display_Diagram_path, const std::string& defect_New_Diagram_path, 
    const std::string& defect_Old_Diagram_path, int margin = 3,
    int pixels_per_cell = 8, int cells_per_block = 6, int orientations = 12, const std::string& metric = "cosine", bool save_flag = false) {
    // 保存原图到指定目录C://defect_display
    if (save_flag) {
        std::string save_dir = "C://defect_display";
        std::filesystem::create_directories(save_dir);
        int save_idx = get_next_save_index(save_dir);
        cv::Mat img1 = imread_unicode(defect_Display_Diagram_path, cv::IMREAD_COLOR);
        cv::Mat img2 = imread_unicode(defect_Old_Diagram_path, cv::IMREAD_COLOR);
        if (!img1.empty()) {
            cv::imwrite(save_dir + "/" + std::to_string(save_idx) + "_display.png", img1);
        }
        if (!img2.empty()) {
            cv::imwrite(save_dir + "/" + std::to_string(save_idx) + "_old.png", img2);
        }
    }

    // 先获取红色矩形区域
    cv::Mat display_img = imread_unicode(defect_Display_Diagram_path, cv::IMREAD_COLOR);
    if (display_img.empty()) {
        std::cout << "未找到缺陷显示图: " << defect_Display_Diagram_path << std::endl;
        return 0.0f;
    }
    cv::Rect coords = get_red_rect_inner_coords(display_img, margin);
    if (coords.width == 0 || coords.height == 0) {
        std::cout << "未找到红色矩形框" << std::endl;
        return 0.0f;
    }

    // 裁剪两张图片从原图
    cv::Mat new_img = imread_unicode(defect_Display_Diagram_path, cv::IMREAD_COLOR);
    if (new_img.empty()) {
        std::cout << "未找到新图: " << defect_Display_Diagram_path << std::endl;
        return 0.0f;
    }
    cv::Mat new_crop = crop_img_with_rect(new_img, coords);

    cv::Mat old_img = imread_unicode(defect_Old_Diagram_path, cv::IMREAD_COLOR);
    if (old_img.empty()) {
        std::cout << "未找到旧图: " << defect_Old_Diagram_path << std::endl;
        return 0.0f;
    }
    cv::Mat old_crop = crop_img_with_rect(old_img, coords);

    // 临时保存裁剪结果到临时文件（compute_hog_similarity 只接受路径）
    std::string tmp1 = "tmp_new_crop.png";
    std::string tmp2 = "tmp_old_crop.png";
    cv::imwrite(tmp1, new_crop);
    cv::imwrite(tmp2, old_crop);

    // 计算HOG相似度
    std::optional<float> defect_hog = compute_hog_similarity(tmp1, tmp2, pixels_per_cell, cells_per_block, orientations, metric);
    if (defect_hog) {
        //std::cout << defect_Display_Diagram_path << "缺陷显示图与旧图" << defect_Old_Diagram_path
        //    << "的HOG相似度(pixels_per_cell = 8, cells_per_block = 6, orientations = 12, metric='euclidean'): "
        //    << *defect_hog << std::endl;
        return *defect_hog;
    }
    else {
        std::cout << "HOG相似度计算失败" << std::endl;
        return 0.0f;
    }

    // 删除临时文件
    //std::remove(tmp1.c_str());
    //std::remove(tmp2.c_str());
}

// 带自动保存的基于HOG+LBP相似度的异常区域验证(使用原图扩大范围)
float area_anomaly_calibration_based_hoglbp_with_save(const std::string& defect_Display_Diagram_path, const std::string& defect_New_Diagram_path, 
    const std::string& defect_Old_Diagram_path, int margin = 3,
    int pixels_per_cell = 8, int cells_per_block = 6, int orientations = 12, const std::string& metric = "cosine", bool save_flag = false)
{   // 保存原图到指定目录
    if (save_flag) {
        std::string save_dir = "C://defect_display";
        std::filesystem::create_directories(save_dir);
        int save_idx = get_next_save_index(save_dir);
        cv::Mat img1 = imread_unicode(defect_Display_Diagram_path, cv::IMREAD_COLOR);
        cv::Mat img2 = imread_unicode(defect_Old_Diagram_path, cv::IMREAD_COLOR);
        if (!img1.empty()) {
            cv::imwrite(save_dir + "/" + std::to_string(save_idx) + "_display.png", img1);
        }
        if (!img2.empty()) {
            cv::imwrite(save_dir + "/" + std::to_string(save_idx) + "_old.png", img2);
        }
    }
    // 先获取红色矩形区域
    cv::Mat display_img = imread_unicode(defect_Display_Diagram_path, cv::IMREAD_COLOR);
    if (display_img.empty()) {
        std::cout << "未找到缺陷显示图: " << defect_Display_Diagram_path << std::endl;
        return 0.0f;
    }
    cv::Rect coords = get_red_rect_inner_coords(display_img, margin);
    if (coords.width == 0 || coords.height == 0) {
        std::cout << "未找到红色矩形框" << std::endl;
        return 0.0f;
    }

    // 裁剪两张图片从原图
    cv::Mat new_img = imread_unicode(defect_Display_Diagram_path, cv::IMREAD_COLOR);
    if (new_img.empty()) {
        std::cout << "未找到新图: " << defect_Display_Diagram_path << std::endl;
        return 0.0f;
    }
    cv::Mat new_crop = crop_img_with_rect(new_img, coords);
    cv::Mat old_img = imread_unicode(defect_Old_Diagram_path, cv::IMREAD_COLOR);
    if (old_img.empty()) {
        std::cout << "未找到旧图: " << defect_Old_Diagram_path << std::endl;
        return 0.0f;
    }
    cv::Mat old_crop = crop_img_with_rect(old_img, coords);

    // 临时保存裁剪结果到临时文件（compute_hoglbp_similarity 只接受路径）
    std::string tmp1 = "tmp_display_crop.png";
    std::string tmp2 = "tmp_old_crop.png";
    cv::imwrite(tmp1, new_crop);
    cv::imwrite(tmp2, old_crop);

    // 计算HOG+LBP相似度
    std::optional<float> sim = compute_hoglbp_similarity(tmp1, tmp2, pixels_per_cell, cells_per_block, orientations, metric);
    if (sim) {
        return *sim;
    }
    else {
        std::cout << "HOG+LBP相似度计算失败" << std::endl;
        return 0.0f;
    }
}
