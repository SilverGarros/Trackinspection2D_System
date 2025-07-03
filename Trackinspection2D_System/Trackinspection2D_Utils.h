#pragma once
#include <string>
#ifdef TRACKINSPECTION2D_UTILS_EXPORTS
#define TRACKINSPECTION2D_API __declspec(dllexport)
#else
#define TRACKINSPECTION2D_API __declspec(dllimport)
#endif
extern "C" {
    TRACKINSPECTION2D_API int test_exported_func();
}
namespace Trackinspection2D_Utils {
    TRACKINSPECTION2D_API std::string extract_image_num(const std::string& filename);
    TRACKINSPECTION2D_API bool is_over_file_exist(const std::string& folder);
    TRACKINSPECTION2D_API void mark_folder_over(const std::string& folder);

}