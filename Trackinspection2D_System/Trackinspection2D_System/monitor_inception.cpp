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
using namespace std;
namespace fs = std::filesystem;

const std::wstring TARGET_EXE = L"Inception_main_onnx.exe";
const std::wstring TARGET_PATH = L"D:\\LuHang_System\\Trackinspection2D_System\\x64\\Debug\\Inception_main.exe"; // 替换为实际路径

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
    _setmode(_fileno(stdout), _O_U16TEXT); // 支持wcout中文

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
        std::this_thread::sleep_for(std::chrono::minutes(30));
    }
    return 0;
}