#ifndef E_BTS_PATH_UTILS_H
#define E_BTS_PATH_UTILS_H

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

#include <metavision/sdk/base/utils/timestamp.h>

namespace e_bts {

inline std::string sanitized_filename_component(const std::string &name) {
    std::string result;
    result.reserve(name.size());

    for (const char ch : name) {
        const bool is_alpha = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
        const bool is_digit = ch >= '0' && ch <= '9';
        if (is_alpha || is_digit || ch == '-' || ch == '_' || ch == '.') {
            result.push_back(ch);
        } else {
            result.push_back('_');
        }
    }

    return result.empty() ? "test" : result;
}

inline std::filesystem::path default_csv_path(const std::filesystem::path &raw_path) {
    std::filesystem::path csv_path = raw_path;
    csv_path.replace_extension(".csv");
    return csv_path;
}

inline unsigned long long timestamp_us(Metavision::timestamp timestamp) {
    return static_cast<unsigned long long>(timestamp);
}

} // namespace e_bts

#endif // E_BTS_PATH_UTILS_H
