#include "Trackinspection2D_Utils.h"
#include "pch.h"
#include <regex>

#include <filesystem>
namespace fs = std::filesystem;
#include <fstream>
#include <iostream>
extern "C" {
    int test_exported_func() { return 42; }
}
namespace Trackinspection2D_Utils {

    std::string extract_image_num(const std::string& filename) {
        std::regex re(R"((\d+)_\d+of\d+\.\w+)");
        std::smatch match;
        if (std::regex_match(filename, match, re)) {
            return match[1];
        }
        return "";
    }

    bool is_over_file_exist(const std::string& folder) {
        return fs::exists(folder + "/over");
    }

    void mark_folder_over(const std::string& folder) {
        std::ofstream file(folder + "/over");
        if (!file) {
            std::cerr << "Error: Unable to create 'over' file in folder: " << folder << std::endl;
            return;
        }
        file.close();
    }

}