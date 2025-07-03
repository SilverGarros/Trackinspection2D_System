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
#include <codecvt>
#include <locale>
using namespace std;
namespace fs = std::filesystem;
string SETTING_PATH = "C:\\DataBase2D\\setting.xml";



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

    const std::wstring TARGET_EXE = L"Inception_main_onnx.exe";
    const std::wstring TARGET_PATH = get_exe_path_from_settings();


    // 路径存在性校验
    if (!fs::exists(TARGET_PATH) || !fs::is_regular_file(TARGET_PATH)) {
        std::wcerr << L"错误：目标程序路径不存在，请联系开发人员。" << std::endl;
        std::wcerr << L"Error: Target executable path does not exist. Please contact the developer." << std::endl;
        return 1;
    }
    std::wcout << L"2D 检测程式监视进程启动中" << std::endl;
    std::wcout << L"2D Detecting Processes Monitoring Processes Being Started..." << std::endl;
    while (true) {
        if (!is_process_running(TARGET_EXE)) {
            std::wcout << L"未检测到 " << TARGET_EXE << L" 进程，正在启动中..." << std::endl;
            std::wcout << L"Process " << TARGET_EXE << L" not detected, starting..." << std::endl;
            start_process(TARGET_PATH);
        }
        else {
            std::wcout << TARGET_EXE << L" 正在运行中。" << std::endl;
            std::wcout << TARGET_EXE << L" is running." << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::minutes(1));
    }
    return 0;
}