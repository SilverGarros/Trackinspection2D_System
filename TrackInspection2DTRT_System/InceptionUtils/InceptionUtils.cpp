#include "pch.h"
#include "InceptionUtils.h"

#include <fstream>
#include <filesystem>
#include <Windows.h>

namespace fs = std::filesystem;

namespace InceptionUtils {

    // 辅助：wstring转utf8
    std::string wstring_to_utf8(const std::wstring& wstr) {
        if (wstr.empty()) return std::string();
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(), nullptr, 0, nullptr, nullptr);
        std::string strTo(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, wstr.data(), (int)wstr.size(), &strTo[0], size_needed, nullptr, nullptr);
        return strTo;
    }

    // 辅助：string(utf8)转wstring
    std::wstring string_to_wstring(const std::string& str) {
        if (str.empty()) return std::wstring();
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.data(), (int)str.size(), nullptr, 0);
        std::wstring wstrTo(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, str.data(), (int)str.size(), &wstrTo[0], size_needed);
        return wstrTo;
    }

    std::string utf8_to_gbk(const std::string& utf8_str) {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, nullptr, 0);
        std::wstring wstr(wlen, 0);
        MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, &wstr[0], wlen);

        int glen = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
        std::string gbk_str(glen, 0);
        WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, &gbk_str[0], glen, nullptr, nullptr);

        if (!gbk_str.empty() && gbk_str.back() == '\0') gbk_str.pop_back();
        return gbk_str;
    }

    INCEPTIONUTILS_API std::string extract_image_num(const std::string& filename) {
        auto pos1 = filename.find_last_of("/\\");
        auto pos2 = filename.find_last_of('.');
        std::string name = filename.substr(pos1 == std::string::npos ? 0 : pos1 + 1,
            (pos2 == std::string::npos ? filename.size() : pos2) - (pos1 == std::string::npos ? 0 : pos1 + 1));
        return name;
    }

    INCEPTIONUTILS_API bool is_over_file_exist(const std::string& folder) {
        fs::path over_file = fs::path(folder) / "over";
        return fs::exists(over_file);
    }

    INCEPTIONUTILS_API void mark_folder_over(const std::string& folder) {
        fs::path over_file = fs::path(folder) / "over";
        std::ofstream ofs(over_file.string());
        ofs << "over" << std::endl;
    }
    INCEPTIONUTILS_API cv::Mat imread_unicode(const std::string& path, int flags) {
        try {
            // 首先尝试直接读取（适用于纯英文路径或已经是GBK编码的路径）
            cv::Mat img = cv::imread(path, flags);
            if (!img.empty()) {
                return img;
            }

            // 如果直接读取失败，尝试编码转换
            // 假设输入的path是UTF-8编码，转换为GBK/ANSI
            std::string gbk_path = utf8_to_gbk(path);
            img = cv::imread(gbk_path, flags);
            if (!img.empty()) {
                return img;
            }

            // 如果仍然失败，使用Windows API读取文件到内存，然后用imdecode
            std::wstring wpath = string_to_wstring(path);

            // 打开文件
            HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ,
                NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            if (hFile == INVALID_HANDLE_VALUE) {
                return cv::Mat(); // 返回空Mat
            }

            // 获取文件大小
            DWORD fileSize = GetFileSize(hFile, NULL);
            if (fileSize == INVALID_FILE_SIZE) {
                CloseHandle(hFile);
                return cv::Mat();
            }

            // 读取文件内容到内存
            std::vector<uchar> buffer(fileSize);
            DWORD bytesRead;
            BOOL success = ReadFile(hFile, buffer.data(), fileSize, &bytesRead, NULL);
            CloseHandle(hFile);

            if (!success || bytesRead != fileSize) {
                return cv::Mat();
            }

            // 使用cv::imdecode从内存解码图像
            return cv::imdecode(buffer, flags);

        }
        catch (const std::exception&) {
            return cv::Mat(); // 异常时返回空Mat
        }
    }

    INCEPTIONUTILS_API bool imwrite_unicode(const std::string& path, const cv::Mat& img, const std::vector<int>& params) {
        try {
            // 首先尝试直接写入（适用于纯英文路径或已经是GBK编码的路径）
            if (cv::imwrite(path, img, params)) {
                return true;
            }

            // 如果直接写入失败，尝试编码转换
            // 假设输入的path是UTF-8编码，转换为GBK/ANSI
            std::string gbk_path = utf8_to_gbk(path);
            if (cv::imwrite(gbk_path, img, params)) {
                return true;
            }

            // 如果仍然失败，使用Windows API写入文件
            std::wstring wpath = string_to_wstring(path);

            // 获取文件扩展名以确定编码格式
            std::string ext = path.substr(path.find_last_of('.'));
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            // 使用cv::imencode编码图像到内存
            std::vector<uchar> buffer;
            if (!cv::imencode(ext, img, buffer, params)) {
                return false;
            }

            // 确保目录存在
            std::wstring dir = wpath.substr(0, wpath.find_last_of(L'\\'));
            if (!dir.empty()) {
                std::error_code ec;
                fs::create_directories(fs::path(path).parent_path(), ec);
            }

            // 使用Windows API写入文件
            HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_WRITE, 0,
                NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
            if (hFile == INVALID_HANDLE_VALUE) {
                return false;
            }

            DWORD bytesWritten;
            BOOL success = WriteFile(hFile, buffer.data(), static_cast<DWORD>(buffer.size()),
                &bytesWritten, NULL);
            CloseHandle(hFile);

            return success && (bytesWritten == buffer.size());

        }
        catch (const std::exception&) {
            return false; // 异常时返回false
        }
    }

    // 用于输出中文消息的辅助函数（无需切换模式）
    INCEPTIONUTILS_API void print_message(const std::wstring& wideMsg) {
        // 每次输出前都设置控制台编码为UTF-8
        SetConsoleOutputCP(CP_UTF8);

        // 转换宽字符到UTF-8
        int size_needed = WideCharToMultiByte(CP_UTF8, 0, wideMsg.c_str(), (int)wideMsg.size(), NULL, 0, NULL, NULL);
        std::string utf8Msg(size_needed, 0);
        WideCharToMultiByte(CP_UTF8, 0, wideMsg.c_str(), (int)wideMsg.size(), &utf8Msg[0], size_needed, NULL, NULL);

        // 输出UTF-8字符串
        std::cout << utf8Msg << std::endl;
    }

    // 将std::string转换为std::wstring并打印
    INCEPTIONUTILS_API void print_message(const std::string& msg) {
        // 每次输出前都设置控制台编码为UTF-8
        SetConsoleOutputCP(CP_UTF8);

        // 转换UTF-8到宽字符
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, msg.c_str(), (int)msg.size(), NULL, 0);
        std::wstring wideMsg(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, msg.c_str(), (int)msg.size(), &wideMsg[0], size_needed);

        // 使用宽字符版本打印
        print_message(wideMsg);
    }
}