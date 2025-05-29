#include <iostream>
#define ORT_API_VERSION 15  // 指定使用API版本15
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string> // 用于 std::wstring 和 std::string 转换
using namespace std;
using namespace cv;

int helloonnxruntime() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeTest");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    const char* model_path = "weights/2025-05-20-16-07-ResNet50-epoch_20-best-acc_0.9939.onnx"; // 替换为您的 ONNX 模型路径
    std::wstring w_model_path = std::wstring(model_path, model_path + strlen(model_path)); // 转换为宽字符

    Ort::Session session(env, w_model_path.c_str(), session_options);
    waitKey(0); // 等待按键事件 
    std::cout << "ONNX model loaded successfully!" << std::endl;

    return 0;
}