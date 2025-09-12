#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <xlnt/xlnt.hpp>
#include "sqlite_loader.h"
#include "xml_loader.h"
namespace fs = std::filesystem;
using namespace xlnt;

#ifndef SQLITE_OK
#define SQLITE_OK 0
#endif
#ifndef SQLITE_ROW
#define SQLITE_ROW 100
#endif
#ifndef SQLITE_DONE
#define SQLITE_DONE 101
#endif
#ifndef SQLITE_NULL
#define SQLITE_NULL 5
#endif

// 全局SQLite加载器
SQLiteLoader g_sqliteLoader;

struct DefectRecord {
    int id;
    std::string type;
    std::string camera;
    std::string image_name;
    double x;
    double y;
    double w;
    double h;
    double confidence;
    double area;
    std::string points;
    double points_area;
};

struct CsvRecord {
    int image_number;
    std::string time;
    double speed1;
    long mileage1;
    double speed2;
    long mileage2;
};

std::vector<CsvRecord> read_csv_mileage(const std::string& csv_path) {
    std::vector<CsvRecord> records;
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "无法打开CSV文件: " << csv_path << std::endl;
        return records;
    }

    std::string line;
    // 跳过标题行
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        CsvRecord record;
        char comma;

        iss >> record.image_number >> comma;
        std::getline(iss, record.time, ',');
        iss >> record.speed1 >> comma;
        iss >> record.mileage1 >> comma;
        iss >> record.speed2 >> comma;
        iss >> record.mileage2;

        records.push_back(record);
    }

    return records;
}

int extract_image_number(const std::string& image_name) {
    try {
        size_t underscore_pos = image_name.find('_');
        if (underscore_pos == std::string::npos) {
            return -1;
        }

        std::string num_str = image_name.substr(0, underscore_pos);
        return std::stoi(num_str);
    }
    catch (...) {
        return -1;
    }
}

long get_mileage_for_image(const std::vector<CsvRecord>& records, const std::string& image_name) {
    int image_num = extract_image_number(image_name);
    if (image_num < 0 || image_num >= static_cast<int>(records.size())) {
        return -1;
    }
    return records[image_num].mileage2; // 使用里程2(米)
}

std::string ensure_valid_utf8(const std::string& str) {
    std::string result;
    result.reserve(str.size());

    for (size_t i = 0; i < str.size(); i++) {
        unsigned char c = static_cast<unsigned char>(str[i]);

        // ASCII 字符 (0-127) 是有效的 UTF-8
        if (c <= 0x7F) {
            result.push_back(str[i]);
            continue;
        }

        // 检查多字节 UTF-8 序列
        size_t sequence_length = 0;
        if ((c & 0xE0) == 0xC0) sequence_length = 2;      // 110xxxxx - 2 字节序列
        else if ((c & 0xF0) == 0xE0) sequence_length = 3; // 1110xxxx - 3 字节序列
        else if ((c & 0xF8) == 0xF0) sequence_length = 4; // 11110xxx - 4 字节序列

        // 无效的 UTF-8 起始字节
        if (sequence_length == 0) {
            result.push_back('?'); // 替换为问号
            continue;
        }

        // 检查后续字节是否有效
        bool valid_sequence = true;
        if (i + sequence_length - 1 >= str.size()) {
            valid_sequence = false; // 序列超出字符串范围
        }
        else {
            for (size_t j = 1; j < sequence_length; j++) {
                if ((static_cast<unsigned char>(str[i + j]) & 0xC0) != 0x80) {
                    valid_sequence = false; // 后续字节无效
                    break;
                }
            }
        }

        if (valid_sequence) {
            // 添加整个有效序列
            for (size_t j = 0; j < sequence_length; j++) {
                result.push_back(str[i + j]);
            }
            i += sequence_length - 1; // 跳过已处理的字节
        }
        else {
            result.push_back('?'); // 替换无效序列为问号
        }
    }

    return result;
}

std::vector<DefectRecord> read_defects_from_db(const std::string& db_path) {
    std::vector<DefectRecord> defects;

    if (!g_sqliteLoader.isLoaded()) {
        std::cerr << "SQLite加载器未初始化" << std::endl;
        return defects;
    }

    sqlite3* db;
    if (g_sqliteLoader.sqlite3_open_fn(db_path.c_str(), &db) != SQLITE_OK) {
        std::cerr << "无法打开数据库: " << db_path << std::endl;
        return defects;
    }

    const char* sql = "SELECT ID, DefectType, Camera, ImageName, X, Y, W, H, Confidence, Area, Points, PointsArea FROM result";
    sqlite3_stmt* stmt;

    if (g_sqliteLoader.sqlite3_prepare_v2_fn(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
        while (g_sqliteLoader.sqlite3_step_fn(stmt) == SQLITE_ROW) {
            DefectRecord record;

            record.id = g_sqliteLoader.sqlite3_column_int_fn(stmt, 0);
            record.type = ensure_valid_utf8(reinterpret_cast<const char*>(g_sqliteLoader.sqlite3_column_text_fn(stmt, 1)));
            record.camera = ensure_valid_utf8(reinterpret_cast<const char*>(g_sqliteLoader.sqlite3_column_text_fn(stmt, 2)));
            record.image_name = ensure_valid_utf8(reinterpret_cast<const char*>(g_sqliteLoader.sqlite3_column_text_fn(stmt, 3)));
            record.x = g_sqliteLoader.sqlite3_column_int_fn(stmt, 4);
            record.y = g_sqliteLoader.sqlite3_column_int_fn(stmt, 5);
            record.w = g_sqliteLoader.sqlite3_column_int_fn(stmt, 6);
            record.h = g_sqliteLoader.sqlite3_column_int_fn(stmt, 7);
            record.confidence = g_sqliteLoader.sqlite3_column_double_fn(stmt, 8);
            record.area = g_sqliteLoader.sqlite3_column_double_fn(stmt, 9);

            if (g_sqliteLoader.sqlite3_column_type_fn(stmt, 10) != SQLITE_NULL) {
                record.points = ensure_valid_utf8(reinterpret_cast<const char*>(g_sqliteLoader.sqlite3_column_text_fn(stmt, 10)));
            }

            if (g_sqliteLoader.sqlite3_column_type_fn(stmt, 11) != SQLITE_NULL) {
                record.points_area = g_sqliteLoader.sqlite3_column_double_fn(stmt, 11);
            }
            else {
                record.points_area = -1;
            }

            defects.push_back(record);
        }
    }

    g_sqliteLoader.sqlite3_finalize_fn(stmt);
    g_sqliteLoader.sqlite3_close_fn(db);

    return defects;
}

void insert_image_into_worksheet(worksheet& ws, const fs::path& image_path, int row, int col, const fs::path& output_image_dir) {
    if (!fs::exists(image_path)) {
        std::cerr << "图片不存在: " << image_path << std::endl;
        return;
    }

    try {
        // 确保输出图片目录存在
        if (!fs::exists(output_image_dir)) {
            fs::create_directories(output_image_dir);
        }

        // 读取图片
        cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "无法读取图片: " << image_path << std::endl;
            return;
        }

        // 调整图片大小以适应Excel单元格
        cv::Size new_size(200, 150);
        cv::Mat resized_img;
        cv::resize(img, resized_img, new_size);

        //// 为每个图片生成唯一的文件名
        //std::string output_filename = "defect_" + std::to_string(row) + "_" +
        //    fs::path(image_path).filename().string();
        // 使用纯ASCII字符创建图片文件名
        std::string output_filename = "img_" + std::to_string(row) + "_" +
            std::to_string(static_cast<unsigned int>(std::hash<std::string>{}(image_path.filename().string()) % 1000000)) + ".jpg";
        fs::path output_path = output_image_dir / output_filename;

        // 保存调整大小后的图片到输出目录
        cv::imwrite(output_path.string(), resized_img);

        // 添加超链接到单元格
        ws.cell(row, col).hyperlink(output_path.string());
        ws.cell(row, col).value("查看图片");

        // 设置单元格样式（蓝色和下划线，类似超链接）
        ws.cell(row, col).font().color(xlnt::rgb_color(0, 0, 255));
        ws.cell(row, col).font().underline(xlnt::font::underline_style::single);
    }
    catch (const std::exception& e) {
        std::cerr << "插入图片错误: " << e.what() << std::endl;
    }
}

void generate_excel_for_folder(const fs::path& folder_path, const fs::path& output_dir) {
    try {
        std::string route_name = ensure_valid_utf8(folder_path.filename().string());
        fs::path route_output_dir = output_dir / route_name;
        if (!fs::exists(route_output_dir)) {
            fs::create_directories(route_output_dir);
        }


        // 检查result.db是否存在
        fs::path db_path = folder_path / "result.db";
        if (!fs::exists(db_path)) {
            std::cout << "未找到result.db文件在: " << folder_path << std::endl;
            return;
        }

        // 读取缺陷数据
        std::vector<DefectRecord> defects = read_defects_from_db(db_path.string());
        if (defects.empty()) {
            std::cout << "没有缺陷数据在: " << folder_path << std::endl;
            return;
        }

        // 创建图片子目录
        fs::path image_output_dir = route_output_dir / "images";
        if (!fs::exists(image_output_dir)) {
            fs::create_directories(image_output_dir);
        }

        // 读取里程数据
        fs::path left_csv_path = folder_path / "IMAQ_左相机.csv";
        fs::path right_csv_path = folder_path / "IMAQ_右相机.csv";

        std::vector<CsvRecord> left_mileage = read_csv_mileage(left_csv_path.string());
        std::vector<CsvRecord> right_mileage = read_csv_mileage(right_csv_path.string());

        // 创建Excel工作簿
        workbook wb;
        worksheet ws = wb.active_sheet();
        ws.title("缺陷报告");

        // 设置标题行 - 所有标题使用ASCII字符串，避免编码问题
        ws.cell("A1").value("缺陷ID");
        ws.cell("B1").value("缺陷类型");
        ws.cell("C1").value("相机位置");
        ws.cell("D1").value("缺陷里程(米)");
        ws.cell("E1").value("图片名称");
        ws.cell("F1").value("置信度");
        ws.cell("G1").value("缺陷面积");
        ws.cell("H1").value("缺陷图片");

        // 设置列宽
        ws.column_properties("A").width = 10;
        ws.column_properties("B").width = 15;
        ws.column_properties("C").width = 10;
        ws.column_properties("D").width = 15;
        ws.column_properties("E").width = 25;
        ws.column_properties("F").width = 10;
        ws.column_properties("G").width = 15;
        ws.column_properties("H").width = 30;

        // 填充数据
        int row = 2;
        for (const auto& defect : defects) {
            try {
                // 获取里程
                long mileage = -1;
                if (defect.camera == "L" && !left_mileage.empty()) {
                    mileage = get_mileage_for_image(left_mileage, defect.image_name);
                }
                else if (defect.camera == "R" && !right_mileage.empty()) {
                    mileage = get_mileage_for_image(right_mileage, defect.image_name);
                }

                // 计算缺陷面积
                double defect_area = defect.points_area > 0 ? defect.points_area : defect.area;

                // 写入数据
                ws.cell(row, 1).value(defect.id);
                ws.cell(row, 2).value(defect.type); // 已在读取时处理
                ws.cell(row, 3).value(defect.camera == "L" ? "左相机" : "右相机");
                ws.cell(row, 4).value(static_cast<double>(mileage));
                ws.cell(row, 5).value(defect.image_name); // 已在读取时处理
                ws.cell(row, 6).value(defect.confidence);
                ws.cell(row, 7).value(defect_area);

                // 查找并插入图片
                std::string stretch_folder = defect.camera == "L" ? "左相机_railhead_stretch" : "右相机_railhead_stretch";
                fs::path image_path = folder_path / stretch_folder / defect.image_name;

                // 插入图片到Excel
                insert_image_into_worksheet(ws, image_path, row, 8, image_output_dir);
            }
            catch (const std::exception& e) {
                std::cerr << "处理缺陷记录时出错: " << e.what() << std::endl;
                // 继续处理下一条记录
            }
            row++;
        }

        // 保存Excel文件
        std::string excel_filename = route_name + "_缺陷报告.xlsx";
        fs::path excel_path = route_output_dir / excel_filename;

        try {
            wb.save(excel_path.string());
            std::cout << "已生成Excel文件: " << excel_path << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "保存Excel文件错误: " << e.what() << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "生成Excel时发生异常: " << e.what() << std::endl;
    }
}

int main() {
    try {
        // 从XML获取配置
        const std::string xml_path = "C:\\DataBase2D\\setting.xml";
        std::string img2D_path = XmlLoader::get_value_from_xml(xml_path, "2DDataSetPath", "D://2DImage");

        // 创建输出目录
        fs::path output_dir = fs::path(img2D_path) / "ExcelExport";
        if (!fs::exists(output_dir)) {
            fs::create_directories(output_dir);
        }

        // 正则表达式匹配文件夹
        std::regex folder_regex(R"(((WP|WN)\d+|Fake|Test|2D)+_\d{4}Y\d{2}M\d{2}D\d{2}h\d{2}m\d{2}s)");

        // 遍历所有符合条件的文件夹
        for (const auto& entry : fs::directory_iterator(img2D_path)) {
            if (entry.is_directory()) {
                std::string folder_name = entry.path().filename().string();
                if (std::regex_match(folder_name, folder_regex)) {
                    // 检查是否有over标签
                    if (fs::exists(entry.path() / "over")) {
                        std::cout << "处理文件夹: " << entry.path() << std::endl;
                        generate_excel_for_folder(entry.path(), output_dir);
                    }
                }
            }
        }

        std::cout << "Excel导出完成，所有文件保存在: " << output_dir << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "程序异常: " << e.what() << std::endl;
        return 1;
    }
}