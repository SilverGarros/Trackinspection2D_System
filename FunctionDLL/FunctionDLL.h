#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <optional>

#ifdef FUNCTIONDLL_EXPORTS
#define FUNCTIONDLL_API __declspec(dllexport)
#else
#define FUNCTIONDLL_API __declspec(dllimport)
#endif

// 此类是从 dll 导出的
class FUNCTIONDLL_API CFunctionDLL {
public:
    CFunctionDLL(void);
    // 可在此添加成员方法
};

FUNCTIONDLL_API cv::Rect get_red_rect_inner_coords(const cv::Mat& img, int margin);
FUNCTIONDLL_API cv::Mat crop_img_with_rect(const cv::Mat& img, const cv::Rect& rect);
FUNCTIONDLL_API std::vector<float> calc_hist(const cv::Mat& lbp, int n_bins);
FUNCTIONDLL_API float l2_similarity(const std::vector<float>& h1, const std::vector<float>& h2);
FUNCTIONDLL_API float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2);
FUNCTIONDLL_API std::optional<float> compute_hog_similarity(
    const std::string& img1_path,
    const std::string& img2_path,
    int pixels_per_cell,
    int cells_per_block,
    int orientations,
    const std::string& metric
);
FUNCTIONDLL_API float area_anomaly_calibration(
    const std::string& defect_Display_Diagram_path,
    const std::string& defect_Old_Diagram_path,
    int margin,
    int pixels_per_cell, 
    int cells_per_block, 
    int orientations, 
    const std::string& metric
);
FUNCTIONDLL_API float area_anomaly_calibration_based_hoglbp(
    const std::string& defect_Display_Diagram_path,
    const std::string& defect_Old_Diagram_path,
    int margin,
    int pixels_per_cell,
    int cells_per_block,
    int orientations,
    const std::string& metric
);
FUNCTIONDLL_API float area_anomaly_calibration_with_save(
    const std::string& defect_Display_Diagram_path,
    const std::string& defect_New_Diagram_path,
    const std::string& defect_Old_Diagram_path,
    int margin,
    int pixels_per_cell,
    int cells_per_block,
    int orientations,
    const std::string& metric,
    bool saveflag
);

FUNCTIONDLL_API float area_anomaly_calibration_based_hoglbp_with_save(
    const std::string& defect_Display_Diagram_path,
    const std::string& defect_New_Diagram_path,
    const std::string& defect_Old_Diagram_path,
    int margin,
    int pixels_per_cell,
    int cells_per_block,
    int orientations,
    const std::string& metric,
    bool saveflag
);

