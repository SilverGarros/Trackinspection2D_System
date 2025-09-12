#ifndef XML_LOADER_H
#define XML_LOADER_H

#include <string>
#include <regex>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace XmlLoader {

    /**
     * @brief 从 XML 文件中读取指定 name 的 value 值，未找到时返回默认值
     * @param xml_path XML 文件路径
     * @param name 要查找的 name 属性值
     * @param default_value 未找到时返回的默认值
     * @return 匹配的 value 值或默认值
     */
    inline std::string get_value_from_xml(const std::string& xml_path, const std::string& name, const std::string& default_value = "") {
        // 检查 XML 文件是否存在
        if (!std::filesystem::exists(xml_path)) {
            std::cerr << "Error: XML file not found: " << xml_path << std::endl;
            return default_value;
        }

        try {
            // 读取 XML 文件内容
            std::ifstream file(xml_path);
            if (!file.is_open()) {
                std::cerr << "Error: Unable to open XML file: " << xml_path << std::endl;
                return default_value;
            }

            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            file.close();

            // 使用正则表达式匹配指定 name 的 value
            std::regex pattern(R"(name\s*=\s*\")" + name + R"(\"\s+type\s*=\s*\"string\"\s+value\s*=\s*\"(.*?)\")");
            std::smatch match;
            if (std::regex_search(content, match, pattern) && match.size() > 1) {
                return match.str(1); // 返回匹配的 value 值
            }
            else {
                std::cerr << "Warning: Specified name not found in XML: " << name << ". Returning default value." << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error: Exception occurred while reading XML: " << e.what() << std::endl;
        }

        return default_value; // 如果未找到，返回默认值
    }

    inline bool string_to_bool(const std::string& str) {
        if (str == "true") {
            return true;
        }
        else if (str == "false") {
            return false;
        }
        else {
            throw std::invalid_argument("Invalid boolean string: " + str);
        }
    }
} // namespace XmlLoader

#endif // XML_LOADER_H