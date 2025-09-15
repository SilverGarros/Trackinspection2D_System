#include "Inspection.h"
#ifndef SQLITE_TRANSIENT
#define SQLITE_TRANSIENT ((void(*)(void*)) -1)
#endif
// 外部定义

// 全局SQLite加载器
SQLiteLoader g_sqliteLoader;
// 常量定义
#define IMG_SIZE 256
#define MAX_THREADS 8
std::mutex cout_mutex;
int Source_IMG_X = 1600;
int Source_IMG_Y = 1024;

bool TestModel = false;

DefectResult_with_position calculateDefectPosition(
    const std::optional<std::vector<SensorData>>& csvData,
    const DefectResult& defect,
    int imageHeight)
{
    DefectResult_with_position result;

    // 首先复制所有基础字段
    result.DefectType = defect.DefectType;
    result.Camera = defect.Camera;

    result.ImageName = defect.ImageName;
    result.offset_x = defect.offset_x;
    result.X = defect.X;
    result.Y = defect.Y;
    result.H = defect.H;
    result.W = defect.W;
    result.Confidence = defect.Confidence;
    result.Area = defect.Area;
    result.PointsArea = defect.PointsArea;
    result.Points = defect.Points;

    // 默认将Position设置为-1（错误状态）
    result.Position = -1.0f;

    // 检查CSV数据是否有效
    if (!csvData.has_value() || csvData->empty()) {
        std::cerr << "CSV数据无效或为空" << std::endl;
        return result;
    }

    int imgIndex, part, totalParts;
    if (!parseImageName(defect.ImageName, imgIndex, part, totalParts)) {
        std::cerr << "Invalid ImageName format: " << defect.ImageName << std::endl;
        return result;
    }

    // 查找当前图片编号和下一编号的传感器数据
    const SensorData* currentData = nullptr;
    const SensorData* nextData = nullptr;

    for (const auto& data : *csvData) {
        if (data.index == imgIndex) {
            currentData = &data;
        }
        else if (data.index == imgIndex + 1) {
            nextData = &data;
        }
    }

    if (!currentData) {
        std::cerr << "No sensor data found for image index: " << imgIndex << std::endl;
        return result;
    }

    // 计算里程差（特殊处理相等情况）
    double mileageDiff;
    if (nextData) {
        mileageDiff = (currentData->value2 == nextData->value2) ? 1.0
            : static_cast<double>(nextData->value2 - currentData->value2);
    }
    else {
        // 如果是最后一张图，使用前一张图的差值或默认值1
        const SensorData* prevData = nullptr;
        for (const auto& data : *csvData) {
            if (data.index == imgIndex - 1) {
                prevData = &data;
                break;
            }
        }
        mileageDiff = (prevData && prevData->value2 != currentData->value2)
            ? static_cast<double>(currentData->value2 - prevData->value2)
            : 1.0;
    }

    // 计算基础里程和像素对应里程
    double baseMileage = currentData->value2 + (mileageDiff / totalParts) * (part - 1);

    // 计算最终位置（注意：使用Y坐标作为纵向位置）
    result.Position = static_cast<float>(
        std::round((baseMileage + (defect.Y / imageHeight) * (mileageDiff / totalParts)) * 100) / 100
        );

    return result;
}
void merge_results_to_db(
    std::vector<DefectResult>& results,
    const std::string& db_folder) {

    //// 按照 ImageName 排序
    //std::sort(results.begin(), results.end(), [](const DefectResult& a, const DefectResult& b) {
    //    return a.ImageName < b.ImageName;
    //    });
    // 使用自然排序算法，使得数字按数值大小排序
    std::sort(results.begin(), results.end(), [](const DefectResult& a, const DefectResult& b) {
        // 提取文件名（不包括扩展名）
        auto getNameWithoutExt = [](const std::string& filename) {
            size_t pos = filename.find_last_of('.');
            return (pos != std::string::npos) ? filename.substr(0, pos) : filename;
            };
        std::string filenameA = getNameWithoutExt(a.ImageName);
        std::string filenameB = getNameWithoutExt(b.ImageName);
        // 查找数字部分
        std::regex numRegex("\\d+");
        std::smatch matchA, matchB;
        // 如果两个文件名都包含数字，按数字大小排序
        if (std::regex_search(filenameA, matchA, numRegex) &&
            std::regex_search(filenameB, matchB, numRegex)) {
            int numA = std::stoi(matchA[0]);
            int numB = std::stoi(matchB[0]);
            if (numA != numB) {
                return numA < numB;
            }
        }
        // 如果数字相同或没有数字，按原始文件名排序
        return a.ImageName < b.ImageName;
        });

    std::string db_path = db_folder + R"(\result.db)";
    if (std::filesystem::exists(db_path)) {
        std::filesystem::remove(db_path);
    }
    if (!g_sqliteLoader.isLoaded()) return;
    sqlite3* db;
    char* err_msg = nullptr;
    int rc = g_sqliteLoader.sqlite3_open_fn(db_path.c_str(), &db);
    if (rc != 0) {
        g_sqliteLoader.sqlite3_close_fn(db);
        return;
    }
    const char* create_sql = R"(
        CREATE TABLE IF NOT EXISTS result (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            DefectType TEXT,
            Camera TEXT,
            ImageName TEXT,
            Source_ImageName TEXT,
            Offset_X REAL,
            X REAL,
            Y REAL,
            H REAL,
            W REAL,
            Source_X REAL,
            Source_Y REAL,
            Source_H REAL,
            Source_W REAL,
            Confidence REAL,
            Area REAL,
            Points TEXT,
            Source_Points TEXT,
            PointsArea REAL
        )
    )";
    g_sqliteLoader.sqlite3_exec_fn(db, create_sql, nullptr, nullptr, &err_msg);

    const char* insert_sql = R"(
        INSERT INTO result (
            DefectType, Camera, ImageName, Source_ImageName,
            Offset_X, X, Y, H, W,
            Source_X, Source_Y, Source_H, Source_W,
            Confidence, Area, Points, Source_Points, PointsArea
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
)";

    sqlite3_stmt* stmt;
    rc = g_sqliteLoader.sqlite3_prepare_v2_fn(db, insert_sql, -1, &stmt, nullptr);
    // V0.2.8 Update 开启事务
    g_sqliteLoader.sqlite3_exec_fn(db, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);

    for (const auto& dr : results) {
        int param_index = 1;

        // DefectType
        if (!dr.DefectType.empty()) {
            if (dr.DefectType == "ZC") {
                // 跳过 DefectType 为 "ZC" 的结果
                continue;
            }
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, param_index++, dr.DefectType.c_str(), -1, SQLITE_TRANSIENT);
        }
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Camera
        if (!dr.Camera.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, param_index++, dr.Camera.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // ImageName
        if (!dr.ImageName.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, param_index++, dr.ImageName.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Source_ImageName (新增)
        if (!dr.Source_ImageName.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, param_index++, dr.Source_ImageName.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Offset_X
        if (dr.offset_x != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.offset_x);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // X
        if (dr.X != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.X);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);
        // Y
        if (dr.Y != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.Y);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);
        // H
        if (dr.H != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.H);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);
        // W
        if (dr.W != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.W);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);
        // Source_Y (新增)
        if (dr.Source_Y != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, static_cast<double>(dr.Source_Y));
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Source_H (新增)
        if (dr.Source_H != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, static_cast<double>(dr.Source_H));
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Source_W (新增)
        if (dr.Source_W != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, static_cast<double>(dr.Source_W));
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Confidence
        if (dr.Confidence != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.Confidence);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Area
        if (dr.Area != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.Area);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Points
        if (!dr.Points.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, param_index++, dr.Points.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Source_Points (新增)
        if (!dr.Source_Points.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, param_index++, dr.Source_Points.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // PointsArea
        if (dr.PointsArea > 0.0f && dr.DefectType == "DK") {
            //cout << "PointsArea:" << dr.PointsArea << endl;
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.PointsArea);
        }
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        g_sqliteLoader.sqlite3_step_fn(stmt);
        g_sqliteLoader.sqlite3_reset_fn(stmt);
        g_sqliteLoader.sqlite3_clear_bindings_fn(stmt);
    }

    // V0.2.8 Update 提交事务
    g_sqliteLoader.sqlite3_exec_fn(db, "COMMIT;", nullptr, nullptr, nullptr);
    g_sqliteLoader.sqlite3_finalize_fn(stmt);
    g_sqliteLoader.sqlite3_close_fn(db);
    std::cout << "预测结果已合并并保存到: " << db_path << std::endl;
}

void merge_results_to_db(
    std::vector<DefectResult_with_position>& results,
    const std::string& db_folder) {
    //// 按照 ImageName 排序
    //std::sort(results.begin(), results.end(), [](const DefectResult& a, const DefectResult& b) {
    //    return a.ImageName < b.ImageName;
    //    });
    // 使用自然排序算法，使得数字按数值大小排序
    std::sort(results.begin(), results.end(), [](const DefectResult_with_position& a, const DefectResult_with_position& b) {
        // 提取文件名（不包括扩展名）
        auto getNameWithoutExt = [](const std::string& filename) {
            size_t pos = filename.find_last_of('.');
            return (pos != std::string::npos) ? filename.substr(0, pos) : filename;
            };
        std::string filenameA = getNameWithoutExt(a.ImageName);
        std::string filenameB = getNameWithoutExt(b.ImageName);
        // 查找数字部分
        std::regex numRegex("\\d+");
        std::smatch matchA, matchB;
        // 如果两个文件名都包含数字，按数字大小排序
        if (std::regex_search(filenameA, matchA, numRegex) &&
            std::regex_search(filenameB, matchB, numRegex)) {
            int numA = std::stoi(matchA[0]);
            int numB = std::stoi(matchB[0]);
            if (numA != numB) {
                return numA < numB;
            }
        }
        // 如果数字相同或没有数字，按原始文件名排序
        return a.ImageName < b.ImageName;
        });

    std::string db_path = db_folder + R"(\result.db)";
    if (std::filesystem::exists(db_path)) {
        std::filesystem::remove(db_path);
    }
    if (!g_sqliteLoader.isLoaded()) return;
    sqlite3* db;
    char* err_msg = nullptr;
    int rc = g_sqliteLoader.sqlite3_open_fn(db_path.c_str(), &db);
    if (rc != 0) {
        g_sqliteLoader.sqlite3_close_fn(db);
        return;
    }
    const char* create_sql = R"(
        CREATE TABLE IF NOT EXISTS result (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            DefectType TEXT,
            Camera TEXT,
            ImageName TEXT,
            Source_ImageName TEXT,
            Offset_X REAL,
            X REAL,
            Y REAL,
            H REAL,
            W REAL,
            Source_X REAL,
            Source_Y REAL,
            Source_H REAL,
            Source_W REAL,
            Confidence REAL,
            Area REAL,
            Points TEXT,
            Source_Points TEXT,
            PointsArea REAL,
            Position REAL
        )
    )";
    g_sqliteLoader.sqlite3_exec_fn(db, create_sql, nullptr, nullptr, &err_msg);

    const char* insert_sql = R"(
        INSERT INTO result (
            DefectType, Camera, ImageName, Source_ImageName,
            Offset_X, X, Y, H, W,
            Source_X, Source_Y, Source_H, Source_W,
            Confidence, Area, Points, Source_Points, PointsArea, Position
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt;
    rc = g_sqliteLoader.sqlite3_prepare_v2_fn(db, insert_sql, -1, &stmt, nullptr);
    // V0.2.8 Update 开启事务
    g_sqliteLoader.sqlite3_exec_fn(db, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);
    for (auto& dr : results) {
        int param_index = 1;

        // DefectType
        // Update 0.2.9处理 DefectType 中包含 "-" 的情况,只保留"-"前
        std::string imageNameSuffix = "";
        if (!dr.DefectType.empty() && dr.DefectType.find('-') != std::string::npos) {
            // 分割 DefectType
            size_t dashPos = dr.DefectType.find('-');
            std::string mainType = dr.DefectType.substr(0, dashPos);
            std::string suffix = dr.DefectType.substr(dashPos + 1);
            dr.DefectType = mainType;
            imageNameSuffix = suffix;
        }
        if (!dr.DefectType.empty()) {
            if (dr.DefectType == "ZC") {
                // 跳过 DefectType 为 "ZC" 的结果
                continue;
            }
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, param_index++, dr.DefectType.c_str(), -1, SQLITE_TRANSIENT);
        }
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Camera
        if (!dr.Camera.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, param_index++, dr.Camera.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Update 0.2.9处理 DefectType 中包含 "-" 的情况,将包含ID信息添加到ImageName中，实现对应
        // 导出的缺陷图像路径的图像名称里出现"GK"问题就是在这来的
        if (!imageNameSuffix.empty() && !dr.ImageName.empty() && imageNameSuffix != "GK") {
            size_t dotPos = dr.ImageName.find_last_of('.');
            if (dotPos != std::string::npos) {
                std::string nameWithoutExt = dr.ImageName.substr(0, dotPos);
                std::string extension = dr.ImageName.substr(dotPos);
                dr.ImageName = nameWithoutExt + "_" + imageNameSuffix + extension;
            }
            else {
                dr.ImageName = dr.ImageName + "_" + imageNameSuffix;
            }
        }
        // ImageName
        if (!dr.ImageName.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, param_index++, dr.ImageName.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);
        // Source_ImageName (新增)
        if (!dr.Source_ImageName.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, param_index++, dr.Source_ImageName.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);
        // Offset_X
        if (dr.offset_x != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.offset_x);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);
        // X
        if (dr.X != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.X);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);
        // Y
        if (dr.Y != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.Y);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);
        // H
        if (dr.H != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.H);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);
        // W
        if (dr.W != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.W);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Source_X (新增)
        if (dr.Source_X != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, static_cast<double>(dr.Source_X));
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Source_Y (新增)
        if (dr.Source_Y != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, static_cast<double>(dr.Source_Y));
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Source_H (新增)
        if (dr.Source_H != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, static_cast<double>(dr.Source_H));
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Source_W (新增)
        if (dr.Source_W != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, static_cast<double>(dr.Source_W));
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Confidence
        if (dr.Confidence != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.Confidence);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);
        // Area
        if (dr.Area != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.Area);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);
        // Points
        if (!dr.Points.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, param_index++, dr.Points.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Source_Points (新增)
        if (!dr.Source_Points.empty())
            g_sqliteLoader.sqlite3_bind_text_fn(stmt, param_index++, dr.Source_Points.c_str(), -1, SQLITE_TRANSIENT);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // PointsArea
        if (dr.PointsArea > 0.0f && dr.DefectType == "DK") {
            //cout << "PointsArea:" << dr.PointsArea << endl;
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.PointsArea);
        }
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        // Position
        if (dr.Position != -1)
            g_sqliteLoader.sqlite3_bind_double_fn(stmt, param_index++, dr.Position);
        else
            g_sqliteLoader.sqlite3_bind_null_fn(stmt, param_index++);

        g_sqliteLoader.sqlite3_step_fn(stmt);
        g_sqliteLoader.sqlite3_reset_fn(stmt);
        g_sqliteLoader.sqlite3_clear_bindings_fn(stmt);
    }

    // V2.8.0 Update 提交事务
    g_sqliteLoader.sqlite3_exec_fn(db, "COMMIT;", nullptr, nullptr, nullptr);

    g_sqliteLoader.sqlite3_finalize_fn(stmt);
    g_sqliteLoader.sqlite3_close_fn(db);
    std::cout << "预测结果已合并并保存到: " << db_path << std::endl;
}
// 单图像处理流程，全部用InceptionDLL和InceptionUtils封装
void process_single_image(
    const std::string& img_path,
    const std::string& railhead_output_path,
    const std::string& stretch_output_path,
    Ort::Session& classify_session,
    YOLO12Infer& detector,
    InspectionGD::GD_AnomalyDetector& gd_detector,
    int img_size,
    int crop_threshold,
    int crop_kernel_size,
    int crop_wide,
    bool center_limit,
    int limit_area,
    int stretch_ratio,
    std::vector<DefectResult>& results,
    std::mutex& results_mutex,
    std::atomic<int>& total_pieces_processed,
    const std::string& camera_side)
{
    // 1. 切分轨面
    if (!fs::exists(img_path)) {
        std::cerr << "图像文件不存在: " << img_path << std::endl;
        return;
    }
    cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "无法读取图像: " << img_path << std::endl;
    }
    else {
        Source_IMG_X = img.cols;;
        Source_IMG_Y = img.rows;
    }
    int Source_IMG_H = Source_IMG_Y;
    //Update InceptionDLL_v0.3.0_N + 2 添加轨面提取偏移量 int offset_x
    int offset_x = -1;
    cv::Mat cropped = InceptionDLL::CropRailhead(img_path, offset_x, crop_threshold, crop_kernel_size, crop_wide, center_limit, limit_area);
    //cout << "InceptionDLL::CropRailhead执行完毕,此时的轨面提取偏移量为" << offset_x << endl;
    if (TestModel) {
        cout << "InceptionDLL::CropRailhead执行完毕,此时的轨面提取偏移量为" << offset_x << endl;
    }
    if (cropped.empty()) return;
    // 检查cropped图像尺寸
    if (cropped.empty() || !cropped.data || cropped.rows <= 0 || cropped.cols <= 0) {
        std::cerr << "无效的裁剪图像: " << img_path << std::endl;
        return;
    }

    //// alpha: 对比度 (1.0 不变)，beta: 亮度偏移 (>0 变亮，<0 变暗)
    //double alpha = 2.2;
    //double beta = 50;   // 根据需要调，单位是像素强度

    //cv::Mat cropped_adj;
    //cropped.convertTo(cropped_adj, -1, alpha, beta);
    //cropped = cropped_adj;
    //cv::Mat gray;
    //cv::cvtColor(cropped, gray, cv::COLOR_BGR2GRAY);

    //auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    //cv::Mat gray_eq;
    //clahe->apply(gray, gray_eq);

    //// 直接使用单通道结果
    //cropped = gray_eq;

    std::string cropped_name = fs::path(img_path).filename().string();
    std::string cropped_path = railhead_output_path + "/" + cropped_name;
    fs::create_directories(railhead_output_path);
    InceptionUtils::imwrite_unicode(cropped_path, cropped);

    // 2. 拉伸与分割
    std::vector<std::string> stretch_piece_paths = InceptionDLL::StretchAndSplit(
        cropped, cropped_name, true, stretch_output_path, stretch_ratio);

    // 3. 片段检测
    for (const auto& out_path : stretch_piece_paths) {
        // std::cout << "正在推理...." << out_path << endl;
        //std::string pred_label = InceptionDLL::ClassifyImage(classify_session, out_path, img_size, out_path);
        ClassificationResult ClassifyResult = InceptionDLL::ClassifyImage(classify_session, out_path, img_size, out_path);
        // 分别获取标签和置信度
        std::string pred_label = ClassifyResult.label;
        float confidence = ClassifyResult.confidence;
        int class_id = ClassifyResult.class_id;

        std::string result;

        if (TestModel) {
            cout << out_path << "被ClassifyImage识别为" << pred_label << endl;
        }

        if (pred_label == "DK" || pred_label == "CS" || pred_label == "GF" || pred_label == "DKCS"
            || pred_label == "ZHC") {
            if (TestModel == true) {
                cout << out_path << "被识别为DK,执行DetectImage进程" << endl;
            }
            std::string detection_result = InceptionDLL::DetectImage(detector, out_path, out_path);
            result = detection_result;
            if (TestModel == true) {
                cout << "返回的结果是：" << result;
            }
            // cout << "===临时Debug===返回的结果是：" << result;
        }
        ////Add v0.3.0
        //else if (pred_label == "GD") {
        //    if (TestModel == true) {
        //        cout << out_path << "被识别为GD,执行GD_AnomalyImage进程" << endl;
        //    }
        //    // 获取GD检测结果（返回vector）
        //    std::vector<std::string> gd_results = gd_detector.GD_AnomalyImage(out_path, true);
        //    // 解析结果
        //    std::string classification_A = "GDZC"; // 默认值
        //    std::string gd_analysis = "无详细结果";
        //    if (!gd_results.empty()) {
        //        classification_A = gd_results[0]; // 第一个元素是分类结果
        //        if (gd_results.size() > 1) {
        //            gd_analysis = gd_results[1]; // 第二个元素是详细分析结果
        //        }
        //    }
        //    if (TestModel == true) {
        //        // 输出结果
        //        std::cout << "GD分类结果: " << classification_A << std::endl;
        //        if (gd_results.size() > 1) {
        //            std::cout << "GD详细分析: " << gd_analysis << std::endl;
        //        }
        //    }
        //    // 生成JSON结果
        //    nlohmann::json j_result = nlohmann::json::array({
        //        { {"class_name", classification_A}, {"gd_analysis", gd_analysis} }
        //        });
        //    result = j_result.dump();
        //    if (TestModel == true) {
        //        cout << "GD检测完成，分类结果: " << classification_A << endl;
        //    }
        //}

        //Update v0.3.0
        else if (pred_label == "HF" || pred_label == "GD") {
            // 执行第一个检测
            if (TestModel == true) {
                cout << out_path << "被识别为HF,先执行GD_AnomalyImage进程" << endl;
            }
            std::string detection_result = InceptionDLL::DetectImage(detector, out_path, out_path);
            // 执行第二个检测
            if (TestModel == true) {
                cout << "然后执行GD_AnomalyImage进程" << endl;
            }
            std::vector<std::string> gd_results = gd_detector.GD_AnomalyImage(out_path, true);
            // 解析GD检测结果
            std::string classification_A = "GDZC"; // 默认值
            std::string gd_analysis = "无详细结果";
            if (!gd_results.empty()) {
                classification_A = gd_results[0]; // 第一个元素是分类结果
                if (gd_results.size() > 1) {
                    gd_analysis = gd_results[1]; // 第二个元素是详细分析结果
                }
            }
            if (TestModel == true) {
                cout << "返回的DetectImage结果是：" << detection_result << endl;
                std::cout << "GD分类结果: " << classification_A << std::endl;
                if (gd_results.size() > 1) {
                    std::cout << "GD详细分析: " << gd_analysis << std::endl;
                }
            }
            // 生成两个独立的JSON对象，与其他分支保持一致
            // 注意：这里生成数组，包含两个检测结果
            nlohmann::json j_result = nlohmann::json::array();
            // 解析DetectImage的结果（假设它返回的是有效的JSON数组或对象）
            try {
                nlohmann::json detect_json = nlohmann::json::parse(detection_result);
                if (detect_json.is_array()) {
                    // 如果是数组，直接添加所有元素
                    for (const auto& item : detect_json) {
                        j_result.push_back(item);
                    }
                }
                else {
                    // 如果是单个对象，添加它
                    j_result.push_back(detect_json);
                }
            }
            catch (...) {
                // 如果解析失败，创建一个默认对象
                j_result.push_back({ {"class_name", "HF_DETECT_ERROR"} });
            }
            // 添加GD检测结果
            j_result.push_back({ {"class_name", classification_A}, {"confidence", confidence},{"gd_analysis", gd_analysis} });
            result = j_result.dump();
        }
        else {
            // 只执行分类的情况，将置信度也存入JSON
            if (TestModel == true) {
                cout << out_path << " 分类结果: " << pred_label
                    << ", 置信度: " << confidence
                    << ", ID: " << class_id << endl;
            }
            // 创建包含置信度的JSON对象
            nlohmann::json j_item = {
                {"class_name", pred_label},
                {"confidence", confidence},  // 关键：添加置信度字段
                {"class_id", class_id}       // 可选：添加类别ID
            };
            nlohmann::json j_result = nlohmann::json::array({ j_item });
            result = j_result.dump();

        }
        // 统计处理的图像片段数
        total_pieces_processed++;
        // 解析result字符串为json
        try {
            nlohmann::json j;
            // 兼容单个/多个结果
            if (result.front() == '[') {
                j = nlohmann::json::parse(result);
            }
            else {
                j = nlohmann::json::array({ nlohmann::json::parse(result) });
            }
            for (const auto& item : j) {
                if (TestModel == true) {
                    if (item.contains("class_name") && item["class_name"] == "ZC") {
                        std::cout << "Skipping item with class_name ZC" << std::endl;
                        continue;
                    }
                }
                // 快速调整导入DB中的类别
                // 跳过ZC
                if (item.contains("class_name") && item["class_name"] == "ZC") continue;
                // 跳过YC
                if (item.contains("class_name") && item["class_name"] == "YC") continue;
                // 跳过光带不均
                // if (item.contains("class_name") && item["class_name"] == "GD") continue;
                // 跳过波磨
                //  if (item.contains("class_name") && item["class_name"] == "BM") continue;


                DefectResult dr;
                dr.DefectType = item.value("class_name", "");
                if (item.contains("class_name") && item["class_name"].is_string()) {
                    dr.DefectType = item["class_name"].get<std::string>();
                }
                else {
                    dr.DefectType = "UNKNOWN";  // 或其他默认值
                }
                dr.Camera = camera_side;
                dr.ImageName = fs::path(out_path).filename().string();
                if (item.contains("bbox") && item["bbox"].is_array() && item["bbox"].size() == 4) {
                    dr.X = item["bbox"][0];
                    dr.Y = item["bbox"][1];
                    dr.W = item["bbox"][2];
                    dr.H = item["bbox"][3];
                }
                dr.Confidence = item.value("confidence", -1.0f);
                dr.Area = item.value("area", 0.0f);
                dr.offset_x = offset_x;
                if (item.contains("contours")) {
                    dr.Points = item["contours"].dump();
                }
                else if (item.contains("gd_analysis")) {
                    dr.Points = item["gd_analysis"].dump();
                }
                dr.PointsArea = item.value("area_contour", 0.0f);
                //Add Trackinspection2D_System_v0.3.0_N+5 对DefectResult进行补齐
                DefectResultCompletionUtils::completeSourceInfo(dr, Source_IMG_H, crop_wide);
                std::lock_guard<std::mutex> lock(results_mutex);
                results.push_back(dr);
            }
        }
        catch (const std::exception& e) {
            DefectResult dr;
            dr.DefectType = pred_label;
            dr.Camera = camera_side;
            cout << "出错的 result 内容: " << result << endl;
            cout << out_path << "解析失败   :\n" << pred_label << "\n" << e.what() << endl;
            //解析失败，写入最基本信息
            dr.ImageName = fs::path(out_path).filename().string();
            dr.offset_x = offset_x;
            std::lock_guard<std::mutex> lock(results_mutex);
            results.push_back(dr);
            continue;
        }

    }
}

// 将相机处理封装为函数
void process_camera_images(const std::string& cam,
    const std::string& cam_side,
    const std::string& folder,
    Ort::Session& classify_session,
    YOLO12Infer& detector,
    InspectionGD::GD_AnomalyDetector& gd_detector,
    int img_size,
    int crop_threshold,
    int crop_kernel_size,
    int crop_wide,
    bool center_limit,
    int limit_area,
    int stretch_ratio,
    std::vector<DefectResult>& local_results,
    std::mutex& results_mutex,
    std::atomic<int>& total_images_processed,
    std::atomic<int>& total_pieces_processed,
    const std::string& camera_side,
    bool skip_FirstAndLastImgs_or_not,
    bool mark_over_or_not) {
    std::string cam_folder = folder + "//" + cam;
    std::string railhead_output_path = folder + "//" + cam + "_railhead";
    std::string stretch_output_path = folder + "//" + cam + "_railhead_stretch";

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
            return;
        }


        //当前代码导致左相机数据丢失的原因是：process_camera_images 内部重新定义了 results，没有使用外部传入的 local_results，导致主线程无法收集左相机的结果。
        //删除 process_camera_images 内部的以下两行
        //std::vector<DefectResult> results;
        //std::mutex results_mutex;
        int idx = 0, total = static_cast<int>(image_files.size());
        std::atomic<int> finished_count{ 0 };

        auto cam_start_time = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        const size_t max_threads = MAX_THREADS;

        // 0.2.7Update 添加跳过首尾图片检测逻辑
        // 预先计算需要跳过的文件
        std::unordered_set<std::string> files_to_skip;
        if (skip_FirstAndLastImgs_or_not && !image_files.empty()) {
            bool found_special = false;
            for (const auto& img_path : image_files) {
                std::filesystem::path img_path_obj(img_path);
                std::string filename = img_path_obj.filename().string();
                if (filename == "0000000.jpg" || filename == "000000.jpg" || filename == "00000.jpg" || filename == "0000.jpg" ||
                    filename == "0000000.jpeg" || filename == "000000.jpeg" || filename == "00000.jpeg" || filename == "0000.jpeg") {
                    files_to_skip.insert(img_path);
                    found_special = true;
                }
            }
            if (found_special && !image_files.empty()) {
                files_to_skip.insert(image_files.back());
            }
        }

        for (const auto& img_path : image_files) {
            threads.emplace_back([&, img_path]() {
                // 检测是否跳过该文件（首尾）
                if (skip_FirstAndLastImgs_or_not) {
                    if (files_to_skip.find(img_path) != files_to_skip.end()) {
                        // 跳过文件，但更新计数和进度
                        finished_count++;
                        total_images_processed++;

                        std::lock_guard<std::mutex> lock(cout_mutex);
                        int percent = finished_count * 100 / total;
                        std::cout << "\r[" << cam_folder << "] 处理进度: " << percent << "% (" << finished_count << "/" << total << ")" << std::flush;
                        return;
                    }
                }
                process_single_image(
                    img_path,
                    railhead_output_path,
                    stretch_output_path,
                    classify_session,
                    detector,
                    gd_detector,
                    img_size,
                    crop_threshold,
                    5,
                    crop_wide,
                    center_limit,
                    limit_area,
                    stretch_ratio,
                    local_results,
                    results_mutex,
                    total_pieces_processed,
                    cam_side
                );
                /* void process_single_image(
                    const std::string & img_path,
                    const std::string & railhead_output_path,
                    const std::string & stretch_output_path,
                    Ort::Session & classify_session,
                    YOLO12Infer & detector,
                    int img_size,
                    int crop_threshold,
                    int crop_kernel_size,
                    int crop_wide,
                    bool center_limit,
                    int limit_area,
                    int stretch_ratio,
                    std::vector<DefectResult>&results,
                    std::mutex & results_mutex,
                    int& total_pieces_processed,
                    const std::string & camera_side*/
                finished_count++;
                total_images_processed++;

                std::lock_guard<std::mutex> lock(cout_mutex);
                int percent = finished_count * 100 / total;
                std::cout << "\r[" << cam_folder << "] 处理进度: " << percent << "% (" << finished_count << "/" << total << ")" << std::flush;
                });

            if (threads.size() >= max_threads) {
                for (auto& t : threads) t.join();
                threads.clear();
            }
        }

        for (auto& t : threads) t.join();
        std::cout << "\r[" << cam_folder << "] 处理进度: " << 100 << "% (" << finished_count << "/" << total << ")" << std::flush;
        std::cout << std::endl;
        //if (!local_results.empty()) {
        //    cout << "process_camera_images local_results 不为空++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        //}
        //else {
        //    cout << "process_camera_images local_results 为空++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        //}
        auto cam_end_time = std::chrono::high_resolution_clock::now();
        auto cam_duration = std::chrono::duration_cast<std::chrono::seconds>(cam_end_time - cam_start_time);

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "处理 " << cam << " 图像 " << total << " 张，耗时 "
                << format_duration(cam_duration) << std::endl;
        }

        //merge_results_to_db(local_results, folder);
    }
}

//Add Inspection_v0.3.0.h_n+7
void collectInspectionFolders(const fs::path& base_path,
    std::vector<std::string>& folder_list,
    const std::regex& target_regex,
    const std::regex& parent_regex,
    int max_depth) {

    // 检查当前目录下的文件夹
    for (const auto& entry : fs::directory_iterator(base_path)) {
        if (entry.is_directory()) {
            std::string folder_name = entry.path().filename().string();
            std::string folder_path = entry.path().string();

            // 情况1：直接匹配目标文件夹
            if (std::regex_match(folder_name, target_regex)) {
                //++++===== 文件夹非空非已检测 ====++++
                if (!InceptionUtils::is_over_file_exist(folder_path)) {
                    folder_list.push_back(folder_path);
                    // std::cout << folder_path << " 已检测，跳过。" << std::endl;
                }
            }
            // 情况2：匹配父文件夹，且可以继续深入一层
            else if (max_depth > 1 && std::regex_match(folder_name, parent_regex)) {
                // 递归检查子目录
                collectInspectionFolders(entry.path(), folder_list,
                    target_regex, parent_regex, max_depth - 1);
            }
        }
    }
}