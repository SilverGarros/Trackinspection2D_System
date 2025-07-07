// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 INCEPTIONSERVICEDLL_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// INCEPTIONSERVICEDLL_API 函数视为是从 DLL 导出的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。

#ifdef INCEPTIONSERVICEDLL_EXPORTS
#define INCEPTIONSERVICEDLL_API __declspec(dllexport)
#else
#define INCEPTIONSERVICEDLL_API __declspec(dllimport)
#endif

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <map>
#include <iostream>

// TensorRT headers - 顺序很重要
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

struct DefectResult {
    std::string DefectType;
    std::string Camera;
    std::string ImageName;
    float X = -1, Y = -1, H = -1, W = -1, Confidence = -1, Area = -1, PointsArea = -1;
    std::string Points;
};
struct ClassificationResult {
    int class_id;
    std::string class_name;
    float confidence;

};
struct DetectionResult {
    std::string class_name;
    cv::Rect bbox;
    float confidence;
    int area;
    std::vector<std::vector<cv::Point>> contours;
    float area_contour;
};
struct InceptionResult {
    enum Type { CLASSIFICATION, DETECTION };
    Type result_type;
    // 图像信息
    std::string img_name;
    std::string img_path;

    // 分类结果
    ClassificationResult classificationresult;

    // 检测结果
    std::vector<DetectionResult> detectionresults;

    InceptionResult() : result_type(CLASSIFICATION) {}
};
namespace Inception_TRT_DLL {
    extern const std::unordered_map<int, std::string> classes_lable_map;
    extern const std::vector<std::string> CLASS_NAMES;
    extern const std::map<std::string, cv::Scalar> CLASS_COLORS;
}
// 为了向后兼容
extern const std::vector<std::string> CLASS_NAMES;
extern const std::map<std::string, cv::Scalar> CLASS_COLORS;

INCEPTIONSERVICEDLL_API std::string detection_results_to_string(const std::vector<DetectionResult>& results);

// TensorRT Logger class
class INCEPTIONSERVICEDLL_API  TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};
class INCEPTIONSERVICEDLL_API Classifier_TRT_Infer {
public:
    Classifier_TRT_Infer(const std::string& engine_file,
        cv::Size input_image_size = cv::Size(256, 256));

    ~Classifier_TRT_Infer();

     int infer(const std::string& image_path);
     ClassificationResult predict(const std::string& image_path);
     ClassificationResult predict(cv::Mat image);

private:
    bool loadEngine(const std::string& engine_file);
    cv::Size input_image_size_;

        // TensorRT components
    TRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // CUDA resources
    void* input_device_buffer_;
    void* output_device_buffer_;
    cudaStream_t stream_;

    // Buffer info
    size_t input_size_;
    size_t output_size_;
    
    // Tensor names for new API
    std::string input_tensor_name_;
    std::string output_tensor_name_;

};
class INCEPTIONSERVICEDLL_API YOLO12TRTInfer {
public:
    YOLO12TRTInfer(const std::string& engine_file,
        cv::Size input_image_size = cv::Size(512, 512),
        float confidence_thres = 0.6f,
        float iou_thres = 0.45f);

    ~YOLO12TRTInfer();

    std::vector<DetectionResult> infer(const std::string& image_path);
    std::vector<DetectionResult> infer(const cv::Mat& img);
    std::vector<DetectionResult> inferWithMultiStream(const cv::Mat& img);
    std::string predict(const std::string& image_path, bool visual = false, bool show_score = true, bool show_class = true, bool save_or_not = false);
    std::vector<DetectionResult> predict(cv::Mat& img, bool visual = false, bool show_score = true, bool show_class = true, bool save_or_not = false, std::string img_path="");
    void draw_box(cv::Mat& img, const DetectionResult& res, bool show_score, bool show_class);
    std::vector<DetectionResult> infer_with_optimized_dk(const std::string& image_path);
    std::string predict_with_optimized_dk(const std::string& image_path,
        bool visual, bool show_score, bool show_class, bool save_or_not);

private:
    int getAvailableStream();
    bool loadEngine(const std::string& engine_file);
    cv::Mat letterbox(const cv::Mat& img, float& h_ratio, float& w_ratio);

    std::vector<float> preprocess(const std::string& image_path, cv::Mat& original_img, float& h_ratio, float& w_ratio);
    std::vector<float> preprocess(const cv::Mat img, cv::Mat& original_img, float& h_ratio, float& w_ratio);

    std::vector<DetectionResult> postprocess(const std::vector<float>& output, int rows, int cols, float h_ratio, float w_ratio, const cv::Mat& original_img);
    std::vector<DetectionResult> postprocess_optimized(
        const std::vector<float>& output, int rows, int cols,
        float h_ratio, float w_ratio, const cv::Mat& original_img);


    cv::Size input_image_size_;
    int input_width_;
    int input_height_;
    float confidence_thres_;
    float iou_thres_;

    // TensorRT components
    TRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    static const int NUM_STREAMS = 3;
    cudaStream_t streams_[NUM_STREAMS];
    void* input_buffers_[NUM_STREAMS];
    void* output_buffers_[NUM_STREAMS];
    std::atomic<int> current_stream_id_{ 0 };
    std::mutex stream_mutex_;

    // CUDA resources - 保持向后兼容
    void* input_device_buffer_;
    void* output_device_buffer_;
    cudaStream_t stream_;

    // Buffer info
    size_t input_size_;
    size_t output_size_;
    
    // Tensor names for new API
    std::string input_tensor_name_;
    std::string output_tensor_name_;
    
    // 保留旧的绑定索引以备兼容
    int input_binding_index_;
    int output_binding_index_;
};
class INCEPTIONSERVICEDLL_API InceptionTRT {
public:
    InceptionTRT(const std::string& classification_engine,
        const std::string& detection_engine,
        cv::Size classification_input_size = cv::Size(256, 256),
        cv::Size detection_input_size = cv::Size(512, 512),
        float confidence_thres = 0.6f,
        float iou_thres = 0.45f,
        int stretch_ratio = 2);

    ~InceptionTRT();
    std::vector<InceptionResult> process(const std::string& image_path,
        const int CROP_WIDE = 850,
        const int CROP_THRESHOLD = 100,
        const std::string CENTER_LIMIT = "true",
        const int LIMIT_AREA = 450,
        const std::string& temp_output_path = "temp_stretched",
        bool output_stretched_images = false);
    std::string getResultAsJson(const InceptionResult& result);
    void setStretchRatio(int ratio) { stretch_ratio_ = ratio; }
    void setDetectionThresholds(float confidence, float iou) {
        if (detector_) {
            confidence_thres_ = confidence;
            iou_thres_ = iou;
        }
    }

private:
    std::unique_ptr<Classifier_TRT_Infer> classifier_;
    std::unique_ptr<YOLO12TRTInfer> detector_;

    cv::Size classification_input_size_;
    cv::Size detection_input_size_;
    float confidence_thres_;
    float iou_thres_;
    int stretch_ratio_;

    cv::Mat railHeadAreaCROP(const std::string& image_path, const int CROP_WIDE, const int CROP_THRESHOLD, const std::string CENTER_LIMIT, const int LIMIT_AREA);

    std::vector<cv::Mat> processImageStretching(const std::string& image_path,cv::Mat railHeadArea,const std::string& output_path,bool output_files);

    //ClassificationResult classifyStretchedImagesByPath(const std::vector<std::string>& stretched_images);
    ClassificationResult classifyImage(const cv::Mat& image,const std::string& image_name);

    //std::vector<DetectionResult> detectionStretchedImagesByPath(const std::vector<std::string>& stretched_images_paths);

    std::vector<DetectionResult> detectImage(const cv::Mat& image,const std::string detection_save_path);
    std::vector<DetectionResult> detectImage_with_TestModel_Flag(const cv::Mat& image, const std::string& detection_save_path);
};

namespace Inception_TRT_DLL {

    INCEPTIONSERVICEDLL_API cv::Mat RailheadCropHighlightCenterArea(
        const cv::Mat& img,
        int threshold,
        int kernel_size,
        int crop_wide,
        bool center_limit,
        int limit_area);

    INCEPTIONSERVICEDLL_API cv::Mat CropRailhead(const std::string& img_path, int crop_threshold, int crop_kernel_size, int crop_wide, bool center_limit, int limit_area);

    INCEPTIONSERVICEDLL_API std::vector<cv::Mat> StretchAndSplit(
        const cv::Mat& cropped,
        const std::string& cropped_name,
        const bool& output_or_not,
        const std::string& stretch_output_path,
        int stretch_ratio);
    INCEPTIONSERVICEDLL_API std::vector<std::string> StretchAndSplit_Paths(
        const cv::Mat& cropped,
        const std::string& cropped_name,
        const bool& output_or_not,
        const std::string& stretch_output_path,
        int stretch_ratio);

    INCEPTIONSERVICEDLL_API std::string ClassifierTRT(
        const cv::Mat& img_input
    );

    INCEPTIONSERVICEDLL_API std::string Detectier_TRT(
        YOLO12TRTInfer& detector, const cv::Mat& img_input);

    INCEPTIONSERVICEDLL_API std::string DetectImage(
        YOLO12TRTInfer& detector,
        const std::string& img_path);

} // namespace Inception_TRT_DLL