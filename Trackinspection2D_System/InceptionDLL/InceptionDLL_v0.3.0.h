// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 INCEPTIONDLL_EXPORTS
// 符号编译的。在使用此 DLL 的任何项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// INCEPTIONDLL_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
#pragma once
#ifdef INCEPTIONDLL_EXPORTS
#define INCEPTIONDLL_API __declspec(dllexport)
#else
#define INCEPTIONDLL_API __declspec(dllimport)
#endif
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// Addv0.3.0_N+1 添加分类置信度——定义返回结构体
struct ClassificationResult {
    std::string label;
    float confidence;
    int class_id;
};

struct DefectResult {
    std::string DefectType;
    std::string Camera;
    std::string ImageName;
    std::string Source_ImageName;
    float X = -1, Y = -1, H = -1, W = -1, Confidence = -1, Area = -1, PointsArea = -1;
    int offset_x = -1;
    int Source_X = -1, Source_Y = -1, Source_H = -1, Source_W = -1;
    std::string Points;
    std::string Source_Points;
};
struct DefectResult_with_position {
    std::string DefectType;
    std::string Camera;
    std::string ImageName;
    std::string Source_ImageName;
    float X = -1, Y = -1, H = -1, W = -1, Confidence = -1, Area = -1, PointsArea = -1;
    int offset_x = -1;
    int Source_X = -1, Source_Y = -1, Source_H = -1, Source_W = -1;
    std::string Points;
    std::string Source_Points;
    float Position;
};
namespace InceptionDLL {    
    extern const std::unordered_map<int, std::string> classes_lable_map;
    extern const std::vector<std::string> CLASS_NAMES;
    extern const std::map<std::string, cv::Scalar> CLASS_COLORS;
}

struct DetectionResult {
    std::string class_name;
    int id = 0;
    cv::Rect bbox;
    float confidence;
    int area;
    std::vector<std::vector<cv::Point>> contours;
    float area_contour;
};

extern const std::vector<std::string> CLASS_NAMES;
extern const std::map<std::string, cv::Scalar> CLASS_COLORS;

INCEPTIONDLL_API std::string detection_results_to_string(const std::vector<DetectionResult>& results);
class INCEPTIONDLL_API YOLO12Infer {
public:
    YOLO12Infer(const std::string& onnx_model,
        cv::Size input_image_size = cv::Size(512, 512),
        float confidence_thres = 0.6f,
        float iou_thres = 0.45f,
        bool use_gpu= false);

    std::vector<DetectionResult> infer(const std::string& image_path);
    std::string predict(const std::string& image_path, bool visual = false, bool show_score = true, bool show_class = true, bool save_or_not = false);
    void draw_box(cv::Mat& img, const DetectionResult& res, bool show_score, bool show_class);
    void draw_box(cv::Mat& img, DetectionResult& res, bool show_score, bool show_class,
        std::map<std::string, int>& class_counter, bool save_single_defects,
        const std::string& original_image_path);
    //Add InceptionDLL_v0.3.0_N+4
    //void draw_box_For_draw_box_classes(cv::Mat& img, DetectionResult& res,
    //bool show_score, bool show_class, std::map<std::string, int>& class_counter);
    //// 新增辅助函数：合并检测结果
    //std::vector<DetectionResult>  merge_detection_results_by_class(
    //    const std::vector<DetectionResult>& results);
    //void draw_box_classes(cv::Mat& img, std::vector<DetectionResult>& results,
    //    bool show_score, bool show_class, std::map<std::string, int>& class_counter,
    //    bool save_single_defects, const std::string& original_image_path);

private:
    cv::Mat letterbox(const cv::Mat& img, float& h_ratio, float& w_ratio);
    std::vector<float> preprocess(const std::string& image_path, cv::Mat& original_img, float& h_ratio, float& w_ratio);
    std::vector<DetectionResult> postprocess(const std::vector<float>& output, int rows, int cols, float h_ratio, float w_ratio, const cv::Mat& original_img, const std::string& image_path);
    void save_single_defect_image(const cv::Mat& img, const DetectionResult& res,
        const std::string& original_path, const std::string& label_with_id);

    cv::Size input_image_size_;
    int input_width_;
    int input_height_;
    float confidence_thres_;
    float iou_thres_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
};

namespace InceptionDLL {
    //Update InceptionDLL_v0.3.0_N+2 轨面提取优化（添加轨面提取的x轴偏移量）
    INCEPTIONDLL_API cv::Mat CropRailhead(
        const std::string& img_path,
        int& offset_x,
        int crop_threshold,
        int crop_kernel_size,
        int crop_wide,
        bool center_limit,
        int limit_area = 50);
    INCEPTIONDLL_API cv::Mat RailheadCropHighlightCenterArea(
        const cv::Mat& img,
        int& center_axis_x,
        int threshold,
        int kernel_size,
        int crop_wide,
        bool center_limit,
        int limit_area= 50);
    //Update InceptionDLL_v0.3.0 轨面提取优化
    INCEPTIONDLL_API std::vector<std::string> StretchAndSplit(
        const cv::Mat& cropped,
        const std::string& cropped_name,
        const bool& output_or_not,
        const std::string& stretch_output_path,
        int stretch_ratio);

    //Update v0.3.0_N+1 添加分类置信度——修改分类返回类型为结构体
    //INCEPTIONDLL_API std::string ClassPredictOnnx(
    //    Ort::Session& session, const cv::Mat& img_input, int img_size);
    //INCEPTIONDLL_API std::string ClassifyImage(
    //    Ort::Session& classify_session,
    //    const std::string& img_path,
    //    int img_size,
    //    const std::string& temp_path);
    INCEPTIONDLL_API ClassificationResult ClassPredictOnnx(
        Ort::Session& session, const cv::Mat& img_input, int img_size);
    INCEPTIONDLL_API ClassificationResult ClassifyImage(
        Ort::Session& classify_session,
        const std::string& img_path,
        int img_size,
        const std::string& temp_path);

    INCEPTIONDLL_API std::string DetectionOnnx(
        YOLO12Infer& detector, const cv::Mat& img_input);

    INCEPTIONDLL_API std::string DetectImage(
        YOLO12Infer& detector,
        const std::string& img_path,
        const std::string& temp_path);

} // namespace InceptionDLL

//Add InspectionGD v0.3.0
namespace InspectionGD{
    struct InspectionGD_Config {
        int num_regions = 40;
        int morph_kernel_size = 15;
        double gdgk_threshold = 400.0;
        double extreme_threshold = 50.0;
        double cov_threshold = 0.1;
        double gradient_threshold = 0.15;
        double mad_threshold = 10.0;
        // 光带区域限制选取积分权重
        double w_area = 0.5;
        double w_center = 0.3;
        double w_rect = 0.2;
    };
    struct GD_RegionInfo {
        int label;
        int area;
        double rectangularity;
        double distance;
        double score;
        std::map<std::string, double> score_detail;
    };
    struct GD_AnalysisResult {
        std::map<std::string, std::map<std::string, double>> details;
        std::map<std::string, std::string> summary;
    };
    struct GD_ProcessResult {
        std::string filename;
        bool gdgk_result;
        bool gdbj_result;
        std::vector<int> widths;
        GD_AnalysisResult analysis;
        std::string category;
    };

    // GD检测的逻辑算法函数
    namespace GD_Algorithms {
        INCEPTIONDLL_API cv::Mat GD_LimitTrackMaskRegion(
            const cv::Mat& mask,
            double w_area = 0.5,
            double w_center = 0.2,
            double w_rect = 0.3);
        INCEPTIONDLL_API std::vector<int> GD_CalculateRegionWidths(
            const cv::Mat& img,
            const cv::Mat& mask,
            int num_regions = 50);
        INCEPTIONDLL_API GD_AnalysisResult GD_AnalyzeRegionWidths(
            const std::vector<int>& widths,
            double gdgk_threshold = 400.0,
            double extreme_threshold = 50.0,
            double cov_threshold = 0.1,
            double gradient_threshold = 0.15,
            double mad_threshold = 10.0);
        INCEPTIONDLL_API std::string GD_AnaysisResult(
            bool gdgk_result,
            bool gdbj_result);
        INCEPTIONDLL_API void GD_PrintAnalysisResults(
            const std::string& filename,
            const std::vector<int>& widths,
            bool gdgk_result,
            bool gdbj_result,
            const GD_AnalysisResult& analysis);
    }

    // 主检测器类
    class INCEPTIONDLL_API GD_AnomalyDetector {
    public:
        GD_AnomalyDetector(const InspectionGD_Config& config = InspectionGD_Config());

        //单张图片处理
        std::string GD_AnomalyImage_result(GD_ProcessResult& gd_result);
        std::string GD_AnomalyImage_result_with_details_CN(GD_ProcessResult& gd_result);
        std::string GD_AnomalyImage_result_with_details(GD_ProcessResult& gd_result);
        GD_ProcessResult GD_AnomalyImage(const std::string& image_path);
        GD_ProcessResult GD_AnomalyImage(cv::Mat image);
        std::vector<std::string> GD_AnomalyImage(const std::string& image_path, bool retrun_full_details);

        std::string SaveGD_Image(
            cv::Mat image,
            bool gdgk_result,
            bool gdbj_result,
            const std::string& source_path,
            const std::string& target_folder
        );
    private:
        InspectionGD_Config InspectionGD_Config_;
        cv::Mat mroph_kernel_;

        // 核心处理流程
        cv::Mat preprocessImage(const std::string& image_path);
        cv::Mat createMask(const cv::Mat& gray_image);
        cv::Mat limitTrackMask(const cv::Mat& mask);
        std::vector<int> anomalyImageExtractWidthsOfRegion(const cv::Mat& img, const cv::Mat& mask);
    };





}

//Add Trackinspection2D_System_v0.3.0_N+5 添加namespace DefectResultCompletionUtils
namespace DefectResultCompletionUtils {
    INCEPTIONDLL_API void completeSourceInfo(DefectResult& dr, int originalHeight = -1, int originalWidth = -1);
    INCEPTIONDLL_API void completeSourceInfo(DefectResult_with_position& dr, int originalHeight = -1, int originalWidth = -1);
    // 批量处理函数
    INCEPTIONDLL_API void completeSourceInfoForAll(std::vector<DefectResult>& results, int originalHeight = -1, int originalWidth = -1);
    INCEPTIONDLL_API void completeSourceInfoForAll(std::vector<DefectResult_with_position>& results, int originalHeight = -1, int originalWidth = -1);
}


