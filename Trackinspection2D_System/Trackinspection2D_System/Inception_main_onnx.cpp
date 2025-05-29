#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <regex>
#include <map>
#include <set>
#include <thread>
#include <future>
#include <fstream>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "sqlite_loader.h"
#include <atomic>
#include <onnxruntime_cxx_api.h>
using namespace std;
SQLiteLoader g_sqliteLoader;
namespace fs = std::filesystem;
#define IMG_SIZE 224
#define USE_GPU true


/**
 * @brief 提取图片文件名中的编号部分。
 * @param filename 图片文件名
 * @return 编号字符串，未匹配返回空字符串
 */
std::string extract_image_num(const std::string& filename);

/**
 * @brief 自然排序比较函数，按数字顺序比较字符串。
 * @param a 字符串a
 * @param b 字符串b
 * @return a < b 返回true，否则false
 */
bool natural_compare(const std::string& a, const std::string& b);

/**
 * @brief 使用ONNX模型对单张图片进行预测（兼容旧API，自动获取输入输出名）。
 * @param session ONNX推理Session
 * @param img_path 图片路径
 * @param img_size 输入图片尺寸
 * @return 预测类别索引，失败返回-1
 */
int predict_onnx_old(Ort::Session& session, const std::string& img_path, int img_size);

/**
 * @brief 使用ONNX模型对单张图片进行预测（优先用通用输入输出名，失败时自动回退）。
 * @param session ONNX推理Session
 * @param img_path 图片路径
 * @param img_size 输入图片尺寸
 * @return 预测类别索引，失败返回-1
 */
int predict_onnx(Ort::Session& session, const std::string& img_path, int img_size);

/**
 * @brief 检查指定文件夹下是否存在"over"文件，表示已处理完成。
 * @param folder 文件夹路径
 * @return 存在返回true，否则false
 */
bool is_over_file_exist(const std::string& folder);

/**
 * @brief 在指定文件夹下创建"over"文件，标记处理完成。
 * @param folder 文件夹路径
 */
void mark_folder_over(const std::string& folder);

/**
 * @brief 对指定文件夹下所有图片进行多线程批量预测，并将结果合并写入数据库。
 * @param folder_path 图片文件夹路径
 * @param session ONNX推理Session
 * @param img_size 输入图片尺寸
 * @param db_folder 数据库文件夹路径
 * @param side 相机侧别（"L"或"R"）
 */
void batch_predict_and_merge(const std::string& folder_path, Ort::Session& session, int img_size, const std::string& db_folder, const std::string& side);

/**
 * @brief 主程序入口。解析参数，初始化ONNX环境，批量检测2D图片并合并结果。
 * @param argc 参数个数
 * @param argv 参数数组
 * @return 0表示成功，其他为失败
 */
int main(int argc, char* argv[]);


// 提取图片编号的函数
std::string extract_image_num(const std::string& filename) {
	std::regex re(R"((\d+)_\d+of\d+\.\w+)");
	std::smatch match;
	if (std::regex_match(filename, match, re)) {
		return match[1];
	}
	return "";
}

// 自然排序比较函数
bool natural_compare(const std::string& a, const std::string& b) {
	try {
		return std::stoi(a) < std::stoi(b);
	}
	catch (std::exception&) {
		return a < b;
	}
}

// 使用ONNX模型进行预测
int predict_onnx_old(Ort::Session& session, const std::string& img_path, int img_size) {
	cv::Mat img = cv::imread(img_path);
	if (img.empty()) {
		std::cerr << "无法读取图像: " << img_path << std::endl;
		return -1;
	}

	cv::resize(img, img, cv::Size(img_size, img_size));
	img.convertTo(img, CV_32F, 1.0 / 255);

	// 转换为正确的输入格式 (NCHW)
	std::vector<float> input_tensor_values;
	input_tensor_values.reserve(img_size * img_size * 3);

	// OpenCV使用BGR，而大多数模型期望RGB
	cv::Mat channels[3];
	cv::split(img, channels);

	// BGR转为RGB并展平到一维数组
	for (int c = 2; c >= 0; c--) {
		const float* channel_data = channels[c].ptr<float>();
		input_tensor_values.insert(input_tensor_values.end(), channel_data, channel_data + img_size * img_size);
	}

	// 创建输入tensor
	std::vector<int64_t> input_shape = { 1, 3, img_size, img_size };
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(),
		input_shape.data(), input_shape.size());

	// 输入输出节点名称
	//旧版本ONNX 已经弃用
	//const char* input_names[] = { session.GetInputName(0, Ort::AllocatorWithDefaultOptions()) };
	//const char* output_names[] = { session.GetOutputName(0, Ort::AllocatorWithDefaultOptions()) };
	//新版本ONNX
	// 使用C API获取输入输出名称
	Ort::AllocatorWithDefaultOptions allocator;

	// 获取第一个输入和输出的名称
	const OrtApi& ort_api = Ort::GetApi();
	const OrtSession* session_ptr = session;

	char* input_name = nullptr;
	char* output_name = nullptr;

	Ort::ThrowOnError(ort_api.SessionGetInputName(session_ptr, 0, allocator, &input_name));
	Ort::ThrowOnError(ort_api.SessionGetOutputName(session_ptr, 0, allocator, &output_name));

	const char* input_names[] = { input_name };
	const char* output_names[] = { output_name };

	// 运行推理
	auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);

	// 释放分配的名称内存
	allocator.Free(input_name);
	allocator.Free(output_name);

	// 获取结果
	const float* output_data = output_tensors[0].GetTensorData<float>();
	size_t output_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

	// 找到最大值的索引
	int max_index = 0;
	float max_value = output_data[0];
	for (size_t i = 1; i < output_count; ++i) {
		if (output_data[i] > max_value) {
			max_value = output_data[i];
			max_index = static_cast<int>(i);
		}
	}

	return max_index;
}
// 使用ONNX模型进行预测
int predict_onnx(Ort::Session& session, const std::string& img_path, int img_size) {
	cv::Mat img = cv::imread(img_path);
	if (img.empty()) {
		std::cerr << "无法读取图像: " << img_path << std::endl;
		return -1;
	}

	cv::resize(img, img, cv::Size(img_size, img_size));
	img.convertTo(img, CV_32F, 1.0 / 255);

	// 转换为正确的输入格式 (NCHW)
	std::vector<float> input_tensor_values;
	input_tensor_values.reserve(img_size * img_size * 3);

	// OpenCV使用BGR，而大多数模型期望RGB
	cv::Mat channels[3];
	cv::split(img, channels);

	// BGR转为RGB并展平到一维数组
	for (int c = 2; c >= 0; c--) {
		const float* channel_data = channels[c].ptr<float>();
		input_tensor_values.insert(input_tensor_values.end(), channel_data, channel_data + img_size * img_size);
	}

	// 创建输入tensor
	std::vector<int64_t> input_shape = { 1, 3, img_size, img_size };
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(),
		input_shape.data(), input_shape.size());

	// 使用直接指定的名称（通用解决方案）
	const char* input_names[] = { "input" };  // 大多数模型使用"input"或"data"
	const char* output_names[] = { "output" }; // 大多数模型使用"output"或"logits"

	try {
		// 运行推理
		auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);

		// 获取结果
		const float* output_data = output_tensors[0].GetTensorData<float>();
		size_t output_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

		// 找到最大值的索引
		int max_index = 0;
		float max_value = output_data[0];
		for (size_t i = 1; i < output_count; ++i) {
			if (output_data[i] > max_value) {
				max_value = output_data[i];
				max_index = static_cast<int>(i);
			}
		}

		return max_index;
	}
	catch (const Ort::Exception& e) {
		// 如果使用默认名称失败，尝试使用节点索引方法
		std::cerr << "尝试默认输入/输出名称失败: " << e.what() << std::endl;
		std::cerr << "尝试获取实际节点名称..." << std::endl;

		// 方法2：尝试通过索引获取节点名称
		try {
			size_t num_input_nodes = session.GetInputCount();
			size_t num_output_nodes = session.GetOutputCount();

			if (num_input_nodes > 0 && num_output_nodes > 0) {
				Ort::AllocatorWithDefaultOptions allocator;
				auto input_name = session.GetInputNameAllocated(0, allocator);
				auto output_name = session.GetOutputNameAllocated(0, allocator);

				const char* actual_input_names[] = { input_name.get() };
				const char* actual_output_names[] = { output_name.get() };

				auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
					actual_input_names, &input_tensor, 1,
					actual_output_names, 1);

				// 获取结果
				const float* output_data = output_tensors[0].GetTensorData<float>();
				size_t output_count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

				int max_index = 0;
				float max_value = output_data[0];
				for (size_t i = 1; i < output_count; ++i) {
					if (output_data[i] > max_value) {
						max_value = output_data[i];
						max_index = static_cast<int>(i);
					}
				}

				return max_index;
			}
		}
		catch (const Ort::Exception& inner_e) {
			std::cerr << "获取节点名称失败: " << inner_e.what() << std::endl;
		}
	}

	std::cerr << "模型推理失败" << std::endl;
	return -1;
}
// 检查over文件是否存在
bool is_over_file_exist(const std::string& folder) {
	return fs::exists(folder + "/over");
}

// 标记文件夹处理完成
// 标记文件夹处理完成
void mark_folder_over(const std::string& folder) {
	std::ofstream file(folder + "/over");
	file.close();
}

// 多线程批量预测并合并结果到数据库
void batch_predict_and_merge(const std::string& folder_path, Ort::Session& session, int img_size, const std::string& db_folder, const std::string& side) {
	std::vector<std::string> image_files;
	for (const auto& entry : fs::directory_iterator(folder_path)) {
		if (entry.is_regular_file()) {
			std::string ext = entry.path().extension().string();
			std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
			if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
				image_files.push_back(entry.path().string());
		}
	}

	std::map<std::string, std::vector<int>> results;
	std::mutex results_mutex;
	std::vector<std::future<void>> futures;

	std::atomic<int> progress_count(0);
	int total = static_cast<int>(image_files.size());
	// 并行处理图像
	for (const auto& img_path : image_files) {
		futures.push_back(std::async(std::launch::async, [&]() {
			int pred = predict_onnx(session, img_path, img_size);
			std::string img_name = fs::path(img_path).filename().string();
			std::string image_num = extract_image_num(img_name);
			if (!image_num.empty()) {
				std::lock_guard<std::mutex> lock(results_mutex);
				results[image_num].push_back(pred);
			}
			// 进度条更新
			int current = ++progress_count;
			if (total > 0) {
				int percent = current * 100 / total;
				// 打印进度条（覆盖同一行）
				std::cout << "\r[" << folder_path << "][" << side << "] 进度: " << percent << "% (" << current << "/" << total << ")" << std::flush;
			}
			}));
	}

	// 等待所有任务完成
	for (auto& f : futures) {
		f.get();
	}
	if (total > 0) std::cout << std::endl; // 换行

	// 合并结果到数据库
	std::string db_path = db_folder + "/result.db";

	// 检查SQLite加载器是否成功加载
	if (!g_sqliteLoader.isLoaded()) {
		std::cerr << "SQLite3 DLL加载失败!" << std::endl;
		return;
	}

	sqlite3* db;
	char* err_msg = nullptr;

	int rc = g_sqliteLoader.sqlite3_open_fn(db_path.c_str(), &db);
	if (rc != 0) {
		std::cerr << "无法打开数据库: " << g_sqliteLoader.sqlite3_errmsg_fn(db) << std::endl;
		g_sqliteLoader.sqlite3_close_fn(db);
		return;
	}

	// 创建表
	const char* create_sql = "CREATE TABLE IF NOT EXISTS result (ImageNum TEXT PRIMARY KEY, L TEXT, R TEXT)";
	rc = g_sqliteLoader.sqlite3_exec_fn(db, create_sql, nullptr, nullptr, &err_msg);
	if (rc != 0) {
		std::cerr << "SQL错误: " << err_msg << std::endl;
		g_sqliteLoader.sqlite3_free_fn(err_msg);
		g_sqliteLoader.sqlite3_close_fn(db);
		return;
	}

	// 查询旧结果
	std::map<std::string, std::pair<std::string, std::string>> old_results;
	const char* select_sql = "SELECT ImageNum, L, R FROM result";
	sqlite3_stmt* stmt;

	rc = g_sqliteLoader.sqlite3_prepare_v2_fn(db, select_sql, -1, &stmt, nullptr);
	if (rc == 0) {
		while (g_sqliteLoader.sqlite3_step_fn(stmt) == 100) { // SQLITE_ROW = 100
			std::string img_num = (const char*)g_sqliteLoader.sqlite3_column_text_fn(stmt, 0);
			std::string l_val = g_sqliteLoader.sqlite3_column_text_fn(stmt, 1) ? (const char*)g_sqliteLoader.sqlite3_column_text_fn(stmt, 1) : "";
			std::string r_val = g_sqliteLoader.sqlite3_column_text_fn(stmt, 2) ? (const char*)g_sqliteLoader.sqlite3_column_text_fn(stmt, 2) : "";
			old_results[img_num] = { l_val, r_val };
		}
	}
	g_sqliteLoader.sqlite3_finalize_fn(stmt);

	// 合并并写入
	g_sqliteLoader.sqlite3_exec_fn(db, "BEGIN TRANSACTION", nullptr, nullptr, &err_msg);

	std::set<std::string> all_image_nums;
	for (const auto& kv : old_results) all_image_nums.insert(kv.first);
	for (const auto& kv : results) all_image_nums.insert(kv.first);

	const char* insert_sql = "INSERT OR REPLACE INTO result (ImageNum, L, R) VALUES (?, ?, ?)";
	rc = g_sqliteLoader.sqlite3_prepare_v2_fn(db, insert_sql, -1, &stmt, nullptr);

	for (const auto& image_num : all_image_nums) {
		std::string l_val = old_results[image_num].first;
		std::string r_val = old_results[image_num].second;

		if (side == "L" && results.find(image_num) != results.end()) {
			std::stringstream ss;
			for (size_t i = 0; i < results[image_num].size(); i++) {
				if (i > 0) ss << ",";
				ss << results[image_num][i];
			}
			l_val = ss.str();
		}
		if (side == "R" && results.find(image_num) != results.end()) {
			std::stringstream ss;
			for (size_t i = 0; i < results[image_num].size(); i++) {
				if (i > 0) ss << ",";
				ss << results[image_num][i];
			}
			r_val = ss.str();
		}
		std::stringstream ss_num;
		ss_num << std::setw(5) << std::setfill('0') << std::stoi(image_num);
		std::string image_num_formatted = ss_num.str();
		g_sqliteLoader.sqlite3_bind_text_fn(stmt, 1, image_num.c_str(), -1, nullptr);
		g_sqliteLoader.sqlite3_bind_text_fn(stmt, 2, l_val.c_str(), -1, nullptr);
		g_sqliteLoader.sqlite3_bind_text_fn(stmt, 3, r_val.c_str(), -1, nullptr);

		rc = g_sqliteLoader.sqlite3_step_fn(stmt);
		if (rc != 101) { // SQLITE_DONE = 101
			std::cerr << "执行SQL失败: " << g_sqliteLoader.sqlite3_errmsg_fn(db) << std::endl;
		}

		g_sqliteLoader.sqlite3_reset_fn(stmt);
		g_sqliteLoader.sqlite3_clear_bindings_fn(stmt);
	}

	g_sqliteLoader.sqlite3_finalize_fn(stmt);
	g_sqliteLoader.sqlite3_exec_fn(db, "COMMIT", nullptr, nullptr, &err_msg);
	g_sqliteLoader.sqlite3_close_fn(db);

	std::cout << "预测结果已合并并保存到: " << db_path << std::endl;
	mark_folder_over(folder_path);
}

int main(int argc, char* argv[]) {
	// 解析参数
	std::string img2D_path = "I://2DImage";
	std::string model_type = "ResNet50";
	int img_size = IMG_SIZE;
	bool use_gpu = USE_GPU;


	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "-img2D_path" && i + 1 < argc) img2D_path = argv[++i];
		else if (arg == "-model_type" && i + 1 < argc) model_type = argv[++i];
		else if (arg == "-use_gpu") use_gpu = true;
	}

	// 设置ONNX Runtime环境
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);

	if (use_gpu) {
#ifdef _WIN32
		OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
		if (status != nullptr) {
			std::cerr << "CUDA Execution Provider 初始化失败" << std::endl;
			return -1;
		}
		std::cout << "已启用GPU(CUDA)推理" << std::endl;
#else
		std::cerr << "当前平台未集成CUDA支持" << std::endl;
		return -1;
#endif
	}
	else {
		std::cout << "使用CPU推理" << std::endl;
	}

	// 加载ONNX模型
	std::string model_path = "weights/2025-05-20-16-07-ResNet50-epoch_20-best-acc_0.9939.onnx";
	//Ort::Session session(env, model_path.c_str(), session_options);
	std::wstring wmodel_path = std::wstring(model_path.begin(), model_path.end());
	Ort::Session session(env, wmodel_path.c_str(), session_options);
	cout << "model_path:" << model_path << endl;

	// 路径存在性校验
	if (!fs::exists(img2D_path) || !fs::is_directory(img2D_path)) {
		std::cerr << "2D图像路径异常，请联系开发人员" << std::endl;
		return 0;
	}
	std::vector<std::string> Inspction_folder;
	std::regex folder_regex(R"(2D_\d{14})"); // 匹配2D_+14位数字

	for (const auto& entry : fs::directory_iterator(img2D_path))
	{
		if (entry.is_directory())
		{
			std::string folder_name = entry.path().filename().string();
			if (std::regex_match(folder_name, folder_regex))
			{
				Inspction_folder.push_back(entry.path().string());
			}
		}
	}
	//打印检测到的全部文件夹名称
	std::cout << "检测到的文件夹列表：" << endl;
	for (const auto& folder : Inspction_folder) {
		cout << folder << endl;
	}
	std::cout << "开始检测..." << endl;
	// 处理每个文件夹
	for (const auto& folder : Inspction_folder)
	{
		if (!fs::is_directory(folder)) continue;
		if (is_over_file_exist(folder)) {
			std::cout << folder << " 已检测，跳过。" << std::endl;
			continue;
		}

		for (const auto& cam : { "左相机", "右相机" })
		{
			std::string cam_side = (std::string(cam) == "左相机") ? "L" : "R";
			std::string stretch_output_path = folder + "/" + cam + "_railhead_stretch";
			if (fs::is_directory(stretch_output_path) && !(is_over_file_exist(stretch_output_path))) {
				batch_predict_and_merge(stretch_output_path, session, img_size, folder, cam_side);
			}
		}

		// 标记文件夹已处理
		mark_folder_over(folder);
		std::cout << folder << " 检测完成，已标记 over 。" << std::endl;
	}

	return 0;
}