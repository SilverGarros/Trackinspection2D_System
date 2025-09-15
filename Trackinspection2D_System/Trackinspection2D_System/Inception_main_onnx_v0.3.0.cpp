#include <iostream>
#include <filesystem>
#include <unordered_set> 
#include <vector>
#include <future>
#include <string>
#include <regex>
#include <map>
#include <set>
#include <fstream>
#include <mutex> 
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <onnxruntime_cxx_api.h>
#include "Inspection.h"
#include "sqlite_loader.h"
#include "xml_loader.h"
#include "csv_loader.h"
#include <nlohmann/json.hpp>

#ifndef SQLITE_TRANSIENT
#define SQLITE_TRANSIENT ((void(*)(void*)) -1)
#endif

int main(int argc, char* argv[]) {
    try {
        const std::string xml_path = "C:\\DataBase2D\\setting.xml";
        
        std::string img2D_path = XmlLoader::get_value_from_xml(xml_path, "2DDataSetPath", "D://2DImage");
        std::string database_root = XmlLoader::get_value_from_xml(xml_path, "2DDataBasePath", "C://DataBase2D");
        std::string L_cam_name = XmlLoader::get_value_from_xml(xml_path, "L_cam_name", "左相机");
        std::string R_cam_name = XmlLoader::get_value_from_xml(xml_path, "R_cam_name", "右相机");
        // v2.8.0Update 是否启用是否启用左右相机并线程运行
        bool Doublemutex = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "use_doublemutex_or_not", "false"));
        // v2.7.0Update 是否启用首末图像跳过
        bool skip_FirstAndLastImgs_or_not = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "skip_FirstAndLastImgs_or_not", "true"));
        // 是否创建over标签文件
        bool mark_over_or_not = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "mark_over_or_not", "true"));
        // 是否使用gpu进行运算
        bool use_gpu = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "use_gpu_or_not", "false"));
        // 是否启动数据收集Ut
        bool data_collect_or_not = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "data_collect_or_not", "false"));
        // 读取据收集目标类别（XML中用逗号分隔，如"GD,BM,ZC"）
        std::string collect_target_classes = XmlLoader::get_value_from_xml(
            xml_path, "CollectTargetClasses", "GDZC,GDBJ,BM,DK,CS"
        );
        std::vector<std::string> target_classes = parseClassList(collect_target_classes);
        // 是否启动精确位置计算
        bool calculate_DefectPosition_or_not = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "calculate_DefectPosition_or_not", "false"));

        // 从 XML 文件中获取 CROP_WIDE 和 CROP_THRESHOLD 的值
        int CROP_WIDE = std::stoi(XmlLoader::get_value_from_xml(xml_path, "CROP_WIDE", "660"));
        int CROP_THRESHOLD = std::stoi(XmlLoader::get_value_from_xml(xml_path, "CROP_THRESHOLD", "100"));

        bool center_limit = XmlLoader::string_to_bool(XmlLoader::get_value_from_xml(xml_path, "center_limit", "false"));
        int limit_area = std::stoi(XmlLoader::get_value_from_xml(xml_path, "CROP_THRESHOLD", "250"));
        // 从 XML 文件中获取拉伸比stretch_ratio的值
        int STRETCH_RATIO = std::stoi(XmlLoader::get_value_from_xml(xml_path, "STRETCH_RATIO", "2"));
        // ===== 分类检测模块参数 =====
        // 分类器尺寸
        int CLASSIFER_IMG_SIZE = std::stoi(XmlLoader::get_value_from_xml(xml_path, "CLASSIFIER_SIZE", std::to_string(IMG_SIZE)));
        // 分类器置信度阈值
        int CLASSIFER_CONFIDENCE_THR = std::stof(XmlLoader::get_value_from_xml(xml_path, "CLASSIFIER_CONFIDENCE", "0.5f"));
        // ===== 目标检测模块参数 =====
        // 检测器尺寸
        int DETECTOR_IMG_SIZE = std::stoi(XmlLoader::get_value_from_xml(xml_path, "DETECTOR_SIZE", "512"));
        // 检测器置信度阈值
        float DETECTOR_CONFIDENCE_THR = std::stof(XmlLoader::get_value_from_xml(xml_path, "DETECTOR_CONFIDENCE", "0.5f"));
        float DETECTOR_IOU_THR = std::stof(XmlLoader::get_value_from_xml(xml_path, "DETECTOR_IOU", "0.65f"));

        // 禁用OpenCV日志输出
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
        // 解析命令行参数
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-img2D_path" && i + 1 < argc) img2D_path = argv[++i];
            else if (arg == "-database_root" && i + 1 < argc) database_root = argv[++i];
        }

        std::string classify_model_name = XmlLoader::get_value_from_xml(xml_path, "ClassificationModel", "C1.onnx");
        std::string detect_model_name = XmlLoader::get_value_from_xml(xml_path, "DetectionModel", "D1.onnx");

        std::string model_path_str = database_root + "//weights//" + classify_model_name;
        std::wstring wmodel_path = std::wstring(model_path_str.begin(), model_path_str.end());
        std::string detect_model_path_str = database_root + "//weights//" + detect_model_name;
        std::string detect_model_path = std::string(detect_model_path_str.begin(), detect_model_path_str.end());

        Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetLogSeverityLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL);
        
        if (use_gpu) {
#ifdef _WIN32
            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
#endif
        }
        std::cout << "加载检测模型..." << std::endl;
        Ort::Session classify_session(env, wmodel_path.c_str(), session_options);

        //目标缺陷检测器
        YOLO12Infer detector(detect_model_path, cv::Size(DETECTOR_IMG_SIZE, DETECTOR_IMG_SIZE), DETECTOR_CONFIDENCE_THR, DETECTOR_IOU_THR,use_gpu);
        //Add v0.3.0 GD 检测器参数加载;
        InspectionGD::InspectionGD_Config config;
        InspectionGD::GD_AnomalyDetector gd_detector(config);

        std::cout << "加载模型完成" << std::endl;

        // 路径校验
        if (!fs::exists(img2D_path) || !fs::is_directory(img2D_path)) {
            std::cerr << "2D图像路径异常，请联系开发人员" << std::endl;
            return 0;
        }
        std::vector<std::string> Inspction_folder; // 待检测的文件夹队列
        std::regex folder_regex(R"(((WP|WN)\d+|Fake|Test|2D)+_\d{4}Y\d{2}M\d{2}D\d{2}h\d{2}m\d{2}s)");
        std::regex parent_folder_regex(R"(.+_(Up|Down)_\d{4}Y\d{2}M\d{2}D\d{2}h\d{2}m\d{2}s)");
        //Addv0.3.0_n+7 适配数据采集存放位置外套上下行文件夹后的区间文件夹路径
        //++++===== 递归文件夹处理确定待检测区间文件夹路径（兼容两种模式） ====++++
        collectInspectionFolders(img2D_path, Inspction_folder, folder_regex, parent_folder_regex, 2);
        //++++===== 待检测文件夹处理主循环 ====++++
        for (const auto& folder : Inspction_folder) {
            try {   //Fix v0.3.0 添加try处理，避免某区间出现问题程式中断
                std::cout << "\r\n开始处理线路: " << folder << std::endl;
                if (!fs::is_directory(folder)) continue;
                if (InceptionUtils::is_over_file_exist(folder)) {
                    // std::cout << folder << " 已检测，跳过。" << std::endl;
                    continue;
                }

                // 记录线路区间处理的开始时间
                auto folder_start_time = std::chrono::high_resolution_clock::now();
                std::atomic<int> total_images_processed(0);  // 用于记录线路区间处理的图像总数
                std::atomic<int> total_pieces_processed(0);  // 用于记录线路区间处理的图像片段总数


                if (!Doublemutex) {
                    std::vector<DefectResult_with_position> all_results;

                    std::vector<DefectResult> global_results_left;
                    std::vector<DefectResult> global_results_right;

                    //++++===== 待检测文件夹左右相机处理（单线程）====++++
                    for (const auto& cam : { L_cam_name, R_cam_name }) {
                        std::string cam_side = (std::string(cam) == L_cam_name) ? "L" : "R";
                        std::vector<DefectResult> now_results;
                        std::vector<DefectResult_with_position> now_results_with_position;
                        std::string cam_folder = folder + "//" + cam;
                        std::string railhead_output_path = folder + "//" + cam + "_railhead";
                        std::string stretch_output_path = folder + "//" + cam + "_railhead_stretch";
                        std::string csv_path = folder + "//" + "IMAQ_" + cam + ".csv";
                        bool csv_file_ready = false;
                        auto csv_result = csvSensorData(csv_path);
                        if (csv_result) {
                            csv_file_ready = true;
                        }
                        else {
                            csv_file_ready = false;
                            std::cout << "未能加载" << cam << "csv文件" << std::endl;
                        }
                        if (fs::is_directory(cam_folder)) {
                            fs::create_directories(railhead_output_path);
                            fs::create_directories(stretch_output_path);

                            std::vector<std::string> image_files;
                            for (const auto& entry : fs::directory_iterator(cam_folder)) {
                                if (entry.is_regular_file()) {
                                    std::string ext = entry.path().extension().string();
                                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                                        image_files.push_back(entry.path().string());
                                }
                            }
                            if (image_files.empty()) {
                                std::cout << "\r\n未检测到" << cam << "图像数据，检查" << cam << "是否异常。" << std::endl;
                                continue;
                            }

                            now_results.clear();

                            std::mutex now_results_mutex;
                            int idx = 0, total = static_cast<int>(image_files.size());
                            std::atomic<int> finished_count{ 0 };

                            // 添加相机处理的开始时间
                            auto cam_start_time = std::chrono::high_resolution_clock::now();

                            // 设置多线程
                            std::vector<std::thread> threads;
                            const size_t max_threads = MAX_THREADS;

                            // 启动工作线程
                            for (const auto& img_path : image_files) {
                                threads.emplace_back([&, img_path]() {
                                    process_single_image(
                                        img_path,
                                        railhead_output_path,
                                        stretch_output_path,
                                        classify_session,
                                        detector,
                                        gd_detector,
                                        CLASSIFER_IMG_SIZE,
                                        CROP_THRESHOLD,
                                        5,
                                        CROP_WIDE,
                                        center_limit,
                                        limit_area,
                                        STRETCH_RATIO,
                                        now_results,
                                        now_results_mutex,
                                        total_pieces_processed,
                                        cam_side
                                    );
                                    finished_count++;
                                    total_images_processed++;
                                    {
                                        std::lock_guard<std::mutex> lock(cout_mutex);
                                        int percent = finished_count * 100 / total;
                                        std::cout << "\r[" << cam_folder << "] 顺序处理进度: " << percent << "% (" << finished_count << "/" << total << ")" << std::flush;
                                    }
                                    });

                                if (threads.size() >= max_threads) {
                                    for (auto& t : threads) t.join();
                                    threads.clear();
                                }
                            }
                            std::mutex cout_mutex; // 全局互斥锁
                            for (auto& t : threads) t.join();

                            // 所有图像处理完成后，一次性转换并添加到 all_results
                            if (csv_file_ready) {
                                for (const auto& now_result : now_results) {
                                    DefectResult_with_position result = calculateDefectPosition(csv_result, now_result, 1024);
                                    all_results.push_back(result);
                                }
                            }
                            else {
                                // 如果没有CSV数据，只添加基础结果
                                for (const auto& now_result : now_results) {
                                    DefectResult_with_position result;
                                    // 复制基础字段
                                    result.DefectType = now_result.DefectType;
                                    result.Camera = now_result.Camera;
                                    result.ImageName = now_result.ImageName;
                                    result.offset_x = now_result.offset_x;
                                    result.X = now_result.X;
                                    result.Y = now_result.Y;
                                    result.H = now_result.H;
                                    result.W = now_result.W;
                                    result.Confidence = now_result.Confidence;
                                    result.Area = now_result.Area;
                                    result.PointsArea = now_result.PointsArea;
                                    result.Points = now_result.Points;
                                    result.Position = -1.0f; // 设置为错误状态
                                    all_results.push_back(result);
                                }
                                // 停止进度显示线程
                                std::cout << "\r[" << cam_folder << "] 顺序处理进度: " << 100 << "% (" << finished_count << "/" << total << ")" << std::flush;
                                std::cout << std::endl;

                                // 计算相机处理时间
                                auto cam_end_time = std::chrono::high_resolution_clock::now();
                                auto cam_duration = std::chrono::duration_cast<std::chrono::seconds>(cam_end_time - cam_start_time);

                                {
                                    std::lock_guard<std::mutex> lock(cout_mutex);
                                    std::cout << "处理 " << cam << " 图像 " << total << " 张，耗时 "
                                        << format_duration(cam_duration) << std::endl;
                                }



                            }
                            all_results.insert(all_results.end(), now_results_with_position.begin(), now_results_with_position.end());
                        }
                    }
                    if (!all_results.empty()) {

                        collect_target_images(all_results, folder, target_classes, data_collect_or_not, true, L_cam_name, R_cam_name);
                        merge_results_to_db(all_results, folder);
                    }
                }
                else {
                    //++++===== 待检测文件夹左右相机处理（双线程）====++++

                // 主处理逻辑
                    std::vector<std::future<void>> futures;
                    std::atomic<int> global_total_images_processed{ 0 };
                    std::atomic<int> global_total_pieces_processed{ 0 };
                    std::vector<DefectResult> global_results;
                    std::mutex global_results_mutex;
                    std::vector<DefectResult_with_position> global_results_with_position;

                    // 启动左右相机并行处理
                    for (const auto& cam : { L_cam_name, R_cam_name }) {
                        std::cout << "启动相机处理任务: " << cam << std::endl;
                        futures.push_back(std::async(std::launch::async, [&, cam]() {
                            //std::string cam_name = (cam == L_cam_name) ? "左相机" : "右相机";
                            std::string cam_name = (cam == L_cam_name) ? "DATL" : "DATR";
                            std::string cam_folder = folder + "//" + cam_name;
                            std::string csv_path = folder + "//" + "IMAQ_" + cam_name + ".csv";
                            bool csv_file_ready = false;
                            auto csv_result = csvSensorData(csv_path);
                            //if (csv_result) {
                            //    csv_file_ready = true;
                            //}
                            //else {
                            //    csv_file_ready = false;
                            //    std::cout << "未能加载" << cam_name << "csv文件" << std::endl;
                            //}
                            // 仅当 calculate_DefectPosition_or_not 和 csv_result 均为 true 时，csv_file_ready 才为 true
                            if (calculate_DefectPosition_or_not && csv_result) {
                                csv_file_ready = true;
                            }
                            else {
                                csv_file_ready = false;
                                // 只有 csv_result 为 false 时才提示“未能加载文件”（符合原逻辑的错误提示场景）
                                if (!csv_result) {
                                    std::cout << "未能加载" << cam_name << "csv文件" << std::endl;
                                }
                            }

                            std::vector<DefectResult> local_results;
                            std::vector<DefectResult_with_position> local_results_with_position;
                            std::mutex local_results_mutex;
                            std::string cam_side = (cam == L_cam_name) ? "L" : "R";

                            process_camera_images(
                                cam_name,
                                cam_side,
                                folder,
                                classify_session,
                                detector,
                                gd_detector,
                                CLASSIFER_IMG_SIZE,
                                CROP_THRESHOLD,
                                5,
                                CROP_WIDE,
                                center_limit,
                                limit_area,
                                STRETCH_RATIO,
                                local_results, // 使用局部结果集
                                local_results_mutex,
                                global_total_images_processed,
                                global_total_pieces_processed,
                                cam_side,
                                skip_FirstAndLastImgs_or_not,
                                mark_over_or_not
                            );

                            // 统一计算缺陷详细位置
                            if (csv_file_ready) {
                                for (const auto& result : local_results) {
                                    DefectResult_with_position result_with_pos = calculateDefectPosition(csv_result, result, 1024);
                                    local_results_with_position.push_back(result_with_pos);
                                }
                            }
                            else {
                                // 如果没有CSV数据，只添加基础结果
                                for (const auto& result : local_results) {
                                    DefectResult_with_position result_with_pos;
                                    // 复制基础字段
                                    //Update Trackinspection2D_System_v0.3.0_N+5 对DefectResult进行补齐
                                    result_with_pos.DefectType = result.DefectType;
                                    result_with_pos.Camera = result.Camera;
                                    result_with_pos.ImageName = result.ImageName;
                                    result_with_pos.Source_ImageName = result.Source_ImageName;
                                    result_with_pos.offset_x = result.offset_x;
                                    result_with_pos.X = result.X;
                                    result_with_pos.Y = result.Y;
                                    result_with_pos.H = result.H;
                                    result_with_pos.W = result.W;
                                    result_with_pos.Source_X = result.Source_X;
                                    result_with_pos.Source_Y = result.Source_Y;
                                    result_with_pos.Source_H = result.Source_H;
                                    result_with_pos.Source_W = result.Source_W;

                                    result_with_pos.Confidence = result.Confidence;
                                    result_with_pos.Area = result.Area;
                                    result_with_pos.PointsArea = result.PointsArea;
                                    result_with_pos.Points = result.Points;
                                    result_with_pos.Source_Points = result.Source_Points;
                                    result_with_pos.Position = -1.0f; // 设置为错误状态
                                    local_results_with_position.push_back(result_with_pos);
                                }
                            }

                            std::lock_guard<std::mutex> lock(global_results_mutex);
                            if (TestModel) {
                                if (!local_results.empty()) {
                                    cout << "local_results 不为空++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
                                }
                                else {
                                    cout << "local_results 为空++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
                                }
                            }

                            global_results.insert(global_results.end(), local_results.begin(), local_results.end());

                            global_results_with_position.insert(global_results_with_position.end(),
                                local_results_with_position.begin(),
                                local_results_with_position.end());
                            }));
                    }

                    // 等待所有相机处理完成
                    for (auto& future : futures) {
                        future.get();
                    }
                    if (TestModel) {
                        if (!global_results.empty()) {
                            cout << "global_results 不为空++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
                        }
                        else {
                            cout << "global_results 为空++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
                        }
                    }

                    // 使用带位置的结果进行数据库合并
                    if (!global_results_with_position.empty()) {
                        collect_target_images(global_results_with_position, folder, target_classes, data_collect_or_not, TestModel, L_cam_name, R_cam_name);
                        merge_results_to_db(global_results_with_position, folder);
                    }
                    else if (!global_results.empty()) {
                        // 如果没有带位置的结果但有基础结果，使用基础结果
                        collect_target_images(global_results, folder, target_classes, data_collect_or_not, TestModel, L_cam_name, R_cam_name);
                        merge_results_to_db(global_results, folder);
                    }


                }
                // 计算整个线路处理的总时间
                auto folder_end_time = std::chrono::high_resolution_clock::now();
                auto folder_duration = std::chrono::duration_cast<std::chrono::seconds>(folder_end_time - folder_start_time);

                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "\r\n线路: " << folder << "处理完毕" << std::endl;
                if (total_images_processed > 0) {
                    std::cout << "原始图像总数: " << total_images_processed << " 张" << std::endl;
                }
                if (total_pieces_processed > 0) {
                    std::cout << "拉伸后处理片段总数: " << total_pieces_processed << " 张" << std::endl;
                }
                std::cout << "总耗时: " << format_duration(folder_duration) << std::endl;
                std::cout << "结果保存路径: " << folder << R"(\result.db)" << std::endl;
                if (mark_over_or_not) {
                    InceptionUtils::mark_folder_over(folder);
                }
            }
            catch (const std::exception& e) {
                std::cerr << "\r\n处理文件夹 " << folder << " 时发生异常: " << e.what() << std::endl;
                std::cerr << "跳过该文件夹，继续处理下一个..." << std::endl;
                continue;
            }
            catch (...) {
                std::cerr << "\r\n处理文件夹 " << folder << " 时发生未知异常" << std::endl;
                std::cerr << "跳过该文件夹，继续处理下一个..." << std::endl;
                continue;
            }
        }
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "\r\n当前所有线路检查完成，异常检测进程关闭..." << std::endl;
        std::cout << "All current line anomaly detection are completed, the program is on standby..." << std::endl;
        return 0;
    }catch (const std::exception& e) {
        std::cerr << "程序异常终止: " << e.what() << std::endl;
        return 3;
    }
    catch (...) {
        std::cerr << "程序发生未知异常，异常终止。" << std::endl;
        return 3;
    }
}