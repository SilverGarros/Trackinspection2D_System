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
    inline std::string escapeRegex(const std::string& str) {
        static const std::regex re(R"([.^$|()\\[*+?{}])");
        return std::regex_replace(str, re, R"(\$&)");
    }
    inline void set_value_in_xml(const std::string& xml_path, const std::string& name, const std::string& value) {
        if (!std::filesystem::exists(xml_path)) {
            throw std::runtime_error("XML file not found: " + xml_path);
        }

        std::ifstream file_in(xml_path);
        std::string content((std::istreambuf_iterator<char>(file_in)), std::istreambuf_iterator<char>());
        file_in.close();

        std::string safe_name = escapeRegex(name);
        std::regex pattern(R"(name\s*=\s*\")" + safe_name + R"(\"\s+type\s*=\s*\"string\"\s+value\s*=\s*\")(.*?)\")");

        std::string new_content = std::regex_replace(
            content,
            pattern,
            "name=\"" + name + "\" type=\"string\" value=\"" + value + "\""
        );

        std::ofstream file_out(xml_path, std::ios::trunc);
        file_out << new_content;
        file_out.close();
    }
    inline void set_value_in_xml_2(const std::string& xml_path, const std::string& name, const std::string& value) {
        if (!std::filesystem::exists(xml_path)) {
            throw std::runtime_error("XML file not found: " + xml_path);
        }

        std::ifstream file_in(xml_path);
        std::string content((std::istreambuf_iterator<char>(file_in)), std::istreambuf_iterator<char>());
        file_in.close();

        // 找到目标参数的位置
        std::string target = "name=\"" + name + "\"";
        size_t pos = content.find(target);
        if (pos == std::string::npos) {
            throw std::runtime_error("No param with name=" + name);
        }

        // 在这一行里找 value="..."
        size_t value_pos = content.find("value=\"", pos);
        if (value_pos == std::string::npos) {
            throw std::runtime_error("No value attribute found for " + name);
        }

        value_pos += 7; // 跳过 value="

        size_t end_quote = content.find("\"", value_pos);
        if (end_quote == std::string::npos) {
            throw std::runtime_error("Malformed XML for " + name);
        }

        // 替换 value
        content.replace(value_pos, end_quote - value_pos, value);

        std::ofstream file_out(xml_path, std::ios::trunc);
        file_out << content;
        file_out.close();
    }
    inline void reset_value_in_xml(const std::string& xml_path, const std::string& name, const std::string& default_xml_path) {
        std::string default_value = get_value_from_xml(default_xml_path, name, "");
        set_value_in_xml(xml_path, name, default_value);
    }
    inline void reset_all_in_xml(const std::string& xml_path, const std::string& default_xml_path) {
        if (!std::filesystem::exists(default_xml_path)) {
            throw std::runtime_error("Default XML file not found: " + default_xml_path);
        }

        std::ifstream file_in(default_xml_path);
        std::string default_content((std::istreambuf_iterator<char>(file_in)), std::istreambuf_iterator<char>());
        file_in.close();

        std::ofstream file_out(xml_path, std::ios::trunc);
        file_out << default_content;
        file_out.close();
    }
} // namespace XmlLoader

#endif // XML_LOADER_H