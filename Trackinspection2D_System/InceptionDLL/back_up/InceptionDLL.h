// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 INCEPTIONDLL_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// INCEPTIONDLL_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。

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

struct DefectResult {
    std::string DefectType;
    std::string Camera;
    std::string ImageName;
    float X = -1, Y = -1, H = -1, W = -1, Confidence = -1, Area = -1, PointsArea = -1;
    std::string Points;
};

namespace InceptionDLL {
    extern const std::unordered_map<int, std::string> classes_lable_map;
    extern const std::vector<std::string> CLASS_NAMES;
    extern const std::map<std::string, cv::Scalar> CLASS_COLORS;
}

struct DetectionResult {
    std::string class_name;
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

private:
    cv::Mat letterbox(const cv::Mat& img, float& h_ratio, float& w_ratio);
    std::vector<float> preprocess(const std::string& image_path, cv::Mat& original_img, float& h_ratio, float& w_ratio);
    std::vector<DetectionResult> postprocess(const std::vector<float>& output, int rows, int cols, float h_ratio, float w_ratio, const cv::Mat& original_img);

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

    INCEPTIONDLL_API cv::Mat RailheadCropHighlightCenterArea(
        const cv::Mat& img, 
        int threshold, 
        int kernel_size, 
        int crop_wide,
        bool center_limit,
        int limit_area);
/*        const bool& output_or_not,
        const std::string& crop_output_path*/


    INCEPTIONDLL_API cv::Mat CropRailhead(const std::string& img_path, int crop_threshold, int crop_kernel_size, int crop_wide, bool center_limit, int limit_area);


    INCEPTIONDLL_API std::vector<std::string> StretchAndSplit(
        const cv::Mat& cropped,
        const std::string& cropped_name,
        const bool& output_or_not,
        const std::string& stretch_output_path,
        int stretch_ratio);

    INCEPTIONDLL_API std::string ClassPredictOnnx(
        Ort::Session& session, const cv::Mat& img_input, int img_size);

    INCEPTIONDLL_API std::string ClassifyImage(
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
