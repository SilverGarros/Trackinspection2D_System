#include <windows.h>
#include <tlhelp32.h>
#include <tchar.h>
#include <io.h>
#include <fcntl.h>
#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <sstream>
#include <vector>
#include <codecvt>
#include <locale>
using namespace std;
namespace fs = std::filesystem;
string SETTING_PATH = "C:\\DataBase2D\\setting.xml";

// 从字符串解析时间（格式：HH:MM）
std::pair<int, int> parse_time(const std::string& timeStr) {
    size_t colonPos = timeStr.find(':');
    if (colonPos == std::string::npos) {
        return { -1, -1 }; // 无效格式
    }

    try {
        int hour = std::stoi(timeStr.substr(0, colonPos));
        int minute = std::stoi(timeStr.substr(colonPos + 1));
        return { hour, minute };
    }
    catch (...) {
        return { -1, -1 }; // 转换失败
    }
}
// 获取时间范围配置
std::vector<std::pair<std::string, std::string>> get_time_ranges_from_settings() {
    const std::string SETTINGS_PATH = SETTING_PATH;
    std::vector<std::pair<std::string, std::string>> timeRanges;

    try {
        if (fs::exists(SETTINGS_PATH)) {
            std::ifstream file(SETTINGS_PATH);
            if (file.is_open()) {
                std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                file.close();

                std::regex timePattern(R"(name\s*=\s*\"Gazer_TIME_RANGE_(\d+)\"\s+type\s*=\s*\"string\"\s+value\s*=\s*\"(\d{1,2}:\d{2})-(\d{1,2}:\d{2})\")");
                std::smatch match;

                auto contentBegin = content.cbegin();
                auto contentEnd = content.cend();

                while (std::regex_search(contentBegin, contentEnd, match, timePattern)) {
                    if (match.size() > 3) {
                        timeRanges.push_back({ match.str(2), match.str(3) });
                    }
                    contentBegin = match.suffix().first;
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::wcerr << L"读取时间范围配置时出错: " << e.what() << std::endl;
    }

    return timeRanges;
}

// 从setting.xml文件获取可执行文件路径
std::wstring get_exe_path_from_settings() {
    const std::wstring DEFAULT_PATH = L"C:\\LuHang_System\\Trackinspection2D_System\\Inception_main.exe";
    const std::string SETTINGS_PATH = SETTING_PATH;

    // 检查设置文件是否存在
    if (!fs::exists(SETTINGS_PATH)) {
        std::wcout << L"未找到设置文件: SETTING_PATH，将使用默认路径" << std::endl;
        return DEFAULT_PATH;
    }

    try {
        std::ifstream file(SETTINGS_PATH);
        if (!file.is_open()) {
            std::wcout << L"无法打开设置文件，将使用默认路径" << std::endl;
            return DEFAULT_PATH;
        }

        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        // 使用正则表达式查找路径
        std::regex pathPattern(R"(name\s*=\s*\"TARGET_EXE_PATH\"\s+type\s*=\s*\"string\"\s+value\s*=\s*\"(.*?)\")");
        std::smatch match;
        if (std::regex_search(content, match, pathPattern) && match.size() > 1) {
            // 将提取的路径转换为宽字符串
            std::string pathStr = match.str(1);
            std::wstring path(pathStr.begin(), pathStr.end());

            // 检查提取的路径是否有效
            if (!path.empty() && fs::exists(path)) {
                std::wcout << L"从设置文件加载路径成功: " << path << std::endl;
                return path;
            }
            else {
                std::wcout << L"设置文件中的路径无效，将使用默认路径" << std::endl;
            }
        }
        else {
            std::wcout << L"在设置文件中未找到TARGET_EXE_PATH标签，将使用默认路径" << std::endl;
        }
    }
    catch (const std::exception& e) {
        // 使用 MultiByteToWideChar 替代 std::wstring_convert
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, NULL, 0);
        std::wstring errorMessage(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, &errorMessage[0], size_needed);

        std::wcout << L"读取设置文件时出错: " << errorMessage << L"，将使用默认路径" << std::endl;
    }
        
    // 检查 std::wcout 状态
    if (std::wcout.fail() || std::wcout.bad()) {
        std::wcerr << L"错误：std::wcout 状态异常。" << std::endl;
        std::wcout.clear(); // 尝试清除错误状态
    }

    return DEFAULT_PATH;
}

int get_sleeper_time_from_settings() {
    const int DEFAULT_sleepTime = 20;
    const std::string SETTINGS_PATH = SETTING_PATH;

    // 检查设置文件是否存在
    if (!fs::exists(SETTINGS_PATH)) {
        std::wcout << L"未找到设置文件: SETTING_PATH，将使用默认休眠时间20min" << std::endl;
        return DEFAULT_sleepTime;
    }
    try {
        std::ifstream file(SETTINGS_PATH);
        if (!file.is_open()) {
            std::wcout << L"无法打开设置文件，将使用默认休眠时间20min" << std::endl;
            return DEFAULT_sleepTime;
        }

        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        // 使用正则表达式查找路径
        std::regex pathPattern(R"(name\s*=\s*\"Gazer_Sleep_TIME\"\s+type\s*=\s*\"int\"\s+value\s*=\s*\"(.*?)\")");
        std::smatch match;
        if (std::regex_search(content, match, pathPattern) && match.size() > 1) {
            // 将提取的休眠时间转换为整数
            int sleepTime = std::stoi(match.str(1));

            // 检查提取的时间是否有效
            if (sleepTime > 0) {
                std::wcout << L"加载休眠时间成功: " << sleepTime << L" 分钟" << std::endl;
                return sleepTime;
            }
            else {
                std::wcout << L"设置文件中的休眠时间无效，将使用默认休眠时间" << std::endl;
            }
        }
        else {
            std::wcout << L"在设置文件中未找到TARGET_EXE_PATH标签，将使用默认路径" << std::endl;
            return DEFAULT_sleepTime;
        }

    }
    catch (const std::exception& e) {
        // 使用 MultiByteToWideChar 替代 std::wstring_convert
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, NULL, 0);
        std::wstring errorMessage(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, e.what(), -1, &errorMessage[0], size_needed);

        std::wcout << L"读取设置文件时出错: " << errorMessage << L"，将使用默认休眠时间" << std::endl;
        return DEFAULT_sleepTime;
    }
}

void print_time_range_info(auto timeRanges) {
    if (timeRanges.empty()) {
        std::wcout << L"未找到时间范围配置，使用默认值 / No time range configuration found, using default values" << std::endl;
        timeRanges = { {"10:15", "16:45"}, {"21:15", "05:45"} };
    }

    std::wcout << L"\n=== 可用时间段信息 / Available Time Ranges ===" << std::endl;
    std::wcout << L"总共配置了 " << timeRanges.size() << L" 个时间段 / Total configured time ranges: " << timeRanges.size() << std::endl;

    for (size_t i = 0; i < timeRanges.size(); ++i) {
        auto [startHour, startMinute] = parse_time(timeRanges[i].first);
        auto [endHour, endMinute] = parse_time(timeRanges[i].second);

        std::wcout << L"时间段 " << (i + 1) << L" / Time Range " << (i + 1) << L": "
            << std::setw(2) << std::setfill(L'0') << startHour << L":"
            << std::setw(2) << std::setfill(L'0') << startMinute << L" - "
            << std::setw(2) << std::setfill(L'0') << endHour << L":"
            << std::setw(2) << std::setfill(L'0') << endMinute;

        if (endHour < startHour || (endHour == startHour && endMinute < startMinute)) {
            std::wcout << L" (跨越午夜 / Crosses midnight)";
        }
        std::wcout << std::endl;
    }

    // 获取当前时间
    time_t now = time(nullptr);
    tm localTime;
    localtime_s(&localTime, &now);

    std::wcout << L"当前系统时间 / Current system time: "
        << std::setw(2) << std::setfill(L'0') << localTime.tm_hour << L":"
        << std::setw(2) << std::setfill(L'0') << localTime.tm_min << L":"
        << std::setw(2) << std::setfill(L'0') << localTime.tm_sec << std::endl;

    std::wcout << L"==========================================\n" << std::endl;
}
// 检查当前时间是否在允许的时间范围内
bool is_within_allowed_time(bool print_time_range_info_or_not) {
    const std::string SETTINGS_PATH = SETTING_PATH;

    // 默认时间范围（当配置文件读取失败时使用）
    std::vector<std::pair<std::string, std::string>> defaultTimeRanges = {
        {"10:15", "16:45"},
        {"22:15", "05:45"}
    };

    std::vector<std::pair<std::string, std::string>> timeRanges = get_time_ranges_from_settings();


    // 如果没有成功读取到时间范围，使用默认值
    if (timeRanges.empty()) {
        timeRanges = defaultTimeRanges;
    }
    if (print_time_range_info_or_not) {
        print_time_range_info(timeRanges);
    }
    // 获取当前时间
    time_t now = time(nullptr);
    tm localTime;
    localtime_s(&localTime, &now);

    int currentHour = localTime.tm_hour;
    int currentMinute = localTime.tm_min;

    // 检查当前时间是否在任何允许的时间范围内
    for (const auto& range : timeRanges) {
        auto [startHour, startMinute] = parse_time(range.first);
        auto [endHour, endMinute] = parse_time(range.second);

        if (startHour == -1 || endHour == -1) {
            continue; // 跳过无效的时间范围
        }

        // 处理跨越午夜的时间范围
        if (endHour < startHour || (endHour == startHour && endMinute < startMinute)) {
            // 时间范围跨越午夜（例如 21:15-05:45）
            if ((currentHour > startHour || (currentHour == startHour && currentMinute >= startMinute)) ||
                (currentHour < endHour || (currentHour == endHour && currentMinute <= endMinute))) {
                return true;
            }
        }
        else {
            // 正常时间范围（不跨越午夜）
            if ((currentHour > startHour || (currentHour == startHour && currentMinute >= startMinute)) &&
                (currentHour < endHour || (currentHour == endHour && currentMinute <= endMinute))) {
                return true;
            }
        }
    }

    return false;
}

bool is_process_running(const std::wstring& exe_name) {
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) return false;

    PROCESSENTRY32 pe;
    pe.dwSize = sizeof(PROCESSENTRY32);
    BOOL hasProcess = Process32First(hSnapshot, &pe);
    while (hasProcess) {
        if (_wcsicmp(pe.szExeFile, exe_name.c_str()) == 0) {
            CloseHandle(hSnapshot);
            return true;
        }
        hasProcess = Process32Next(hSnapshot, &pe);
    }
    CloseHandle(hSnapshot);
    return false;
}

void start_process(const std::wstring& exe_path) {
    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    if (CreateProcessW(exe_path.c_str(), NULL, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    } else {
        std::wcerr << L"启动失败: " << exe_path << std::endl;
    }
}

int main() {

    if (_setmode(_fileno(stdout), _O_U16TEXT) == -1) {
        std::wcerr << L"错误：无法设置输出模式为宽字符。" << std::endl;
    }

    const std::wstring TARGET_EXE = L"Inception_main.exe";
    const std::wstring TARGET_PATH = get_exe_path_from_settings();
    const int sleeperTime = get_sleeper_time_from_settings();


    // 路径存在性校验
    if (!fs::exists(TARGET_PATH) || !fs::is_regular_file(TARGET_PATH)) {
        std::wcerr << L"错误：目标程序路径不存在，请联系开发人员。" << std::endl;
        std::wcerr << L"Error: Target executable path does not exist. Please contact the developer." << std::endl;
        return 1;
    }
    std::wcout << L"2D 检测程式监视进程已启动" << std::endl;
    std::wcout << L"2D Detecting Processes Monitoring Processes Being Started..." << std::endl;
    is_within_allowed_time(true);
    while (true) {
        if (is_within_allowed_time(false)) {
            if (!is_process_running(TARGET_EXE)) {
                std::wcout << L"未检测到 " << TARGET_EXE << L" 进程，正在启动中..." << std::endl;
                std::wcout << L"Process " << TARGET_EXE << L" not detected, starting..." << std::endl;
                start_process(TARGET_PATH);
            }
            else {
                std::wcout << TARGET_EXE << L" 正在运行中。\r" << std::endl;
                std::wcout << TARGET_EXE << L" is running.\r" << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::minutes(sleeperTime));
        }
    }
    return 0;
}