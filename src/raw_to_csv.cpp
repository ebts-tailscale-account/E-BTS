#include <filesystem>
#include <iostream>
#include <string>

#include "camera_utils.h"
#include "path_utils.h"

namespace {

void print_usage(const char *program_name) {
    std::cerr << "Usage:\n"
              << "  " << program_name << " /path/to/recording.raw /path/to/[cd_output.csv] /path/to/[trigger_output.csv]\n\n"
              << "Examples:\n(in same folder)\n"
              << "  " << program_name << " recording.raw\n"
              << "  " << program_name << " recording.raw cd.csv\n"
              << "  " << program_name << " recording.raw cd.csv triggers.csv\n";
}

} // namespace

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::filesystem::path input_raw_path = argv[1];
    const std::filesystem::path cd_csv_path =
        argc >= 3 ? std::filesystem::path(argv[2]) : e_bts::default_csv_path(input_raw_path);
    const bool write_triggers = argc >= 4;
    const std::filesystem::path trigger_csv_path = write_triggers ? std::filesystem::path(argv[3]) : std::filesystem::path();

    const auto result = write_triggers ?
                            e_bts::convert_raw_to_csv(input_raw_path, cd_csv_path, &trigger_csv_path) :
                            e_bts::convert_raw_to_csv(input_raw_path, cd_csv_path);
    return result.success ? 0 : 1;
}
