#include <iostream>
#include <filesystem>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

bool matchesFolderPattern(const std::string& folderName) {
    std::regex folder_regex(R"(((WP|WN)\d+|Fake|Test|2D)+_\d{4}Y\d{2}M\d{2}D\d{2}h\d{2}m\d{2}s)");
    return std::regex_match(folderName, folder_regex);
}

void removeFlagFile(const fs::path& folderPath, const std::string& flagName) {
    fs::path flagPath = folderPath / flagName;
    if (fs::exists(flagPath)) {
        try {
            fs::remove(flagPath);
            std::cout << "已删除: " << flagPath.string() << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "删除失败: " << flagPath.string() << " - " << e.what() << std::endl;
        }
    }
}

void processFolder(const fs::path& rootPath, const std::string& flagChoice) {
    if (!fs::exists(rootPath) || !fs::is_directory(rootPath)) {
        std::cerr << "错误: 无效的文件夹路径 " << rootPath.string() << std::endl;
        return;
    }

    int matchedFolders = 0;
    int removedFiles = 0;

    for (const auto& entry : fs::directory_iterator(rootPath)) {
        if (entry.is_directory()) {
            std::string folderName = entry.path().filename().string();
            if (matchesFolderPattern(folderName)) {
                bool folderMatched = false;

                // 处理单个标志文件
                if (flagChoice == "finish" || flagChoice == "over") {
                    fs::path flagPath = entry.path() / flagChoice;
                    if (fs::exists(flagPath)) {
                        if (!folderMatched) {
                            matchedFolders++;
                            folderMatched = true;
                        }
                        removeFlagFile(entry.path(), flagChoice);
                        removedFiles++;
                    }
                }
                // 处理两种标志文件
                else if (flagChoice == "all") {
                    bool filesFound = false;

                    // 检查并删除over文件
                    fs::path overPath = entry.path() / "over";
                    if (fs::exists(overPath)) {
                        filesFound = true;
                        removeFlagFile(entry.path(), "over");
                        removedFiles++;
                    }

                    // 检查并删除finish文件
                    fs::path finishPath = entry.path() / "finish";
                    if (fs::exists(finishPath)) {
                        filesFound = true;
                        removeFlagFile(entry.path(), "finish");
                        removedFiles++;
                    }

                    // 只有当文件夹中有相关标志文件时才计数
                    if (filesFound && !folderMatched) {
                        matchedFolders++;
                    }
                }
            }
        }
    }

    std::cout << "\n处理完成：" << std::endl;
    std::cout << "- 匹配文件夹数量: " << matchedFolders << std::endl;
    std::cout << "- 删除标志文件数量: " << removedFiles << std::endl;
}

int main() {
    bool continueRunning = true;

    while (continueRunning) {
        std::string rootPath;
        std::string flagChoice;

        std::cout << "\n=== 标志文件删除工具 ===" << std::endl;

        // 获取文件夹路径
        std::cout << "请输入根文件夹路径: ";
        std::getline(std::cin, rootPath);

        // 获取标志文件选择
        std::cout << "请选择要删除的标志文件类型:" << std::endl;
        std::cout << "1. finish" << std::endl;
        std::cout << "2. over" << std::endl;
        std::cout << "3. over&finish" << std::endl;
        std::cout << "请输入选项 (1/2/3): ";
        std::getline(std::cin, flagChoice);

        std::string flagName;
        if (flagChoice == "1") {
            flagName = "finish";
        }
        else if (flagChoice == "2") {
            flagName = "over";
        }
        else if (flagChoice == "3") {
            flagName = "all";
        }
        else {
            std::cerr << "错误: 无效的选项" << std::endl;
            continue;
        }

        std::string displayName = (flagName == "all") ? "over 和 finish" : flagName;
        std::cout << "将删除所有符合正则表达式的文件夹中的 '" << displayName << "' 文件" << std::endl;
        std::cout << "该操作不可撤销，请确认操作? (y/n): ";
        std::string confirm;
        std::getline(std::cin, confirm);

        if (confirm == "y" || confirm == "Y") {
            processFolder(rootPath, flagName);
        }
        else {
            std::cout << "操作已取消" << std::endl;
        }

        // 询问用户是否继续运行程序
        std::cout << "\n是否继续运行程序? (y/n): ";
        std::string continueChoice;
        std::getline(std::cin, continueChoice);

        continueRunning = (continueChoice == "y" || continueChoice == "Y");
    }

    std::cout << "程序已退出，按任意键关闭窗口...";
    std::cin.get();
    return 0;
}