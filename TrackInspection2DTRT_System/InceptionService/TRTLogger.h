// TRTLogger.h
#pragma once
#include <iostream>
#include <NvInfer.h>
using namespace std;
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};