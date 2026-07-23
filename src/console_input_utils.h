#ifndef E_BTS_CONSOLE_INPUT_UTILS_H
#define E_BTS_CONSOLE_INPUT_UTILS_H

#include <iostream>
#include <sstream>
#include <string>

namespace e_bts {

// Reads one line from stdin. An empty line (just Enter) yields default_value.
// Otherwise the line must parse as exactly one Value with no trailing content.
// Shared by every interactive numeric prompt (accumulation time, event
// collection time, circle density, ...) so parsing behavior stays consistent.
template<typename Value>
bool read_value_or_default(Value default_value, Value &value) {
    std::string input_line;
    if (!std::getline(std::cin, input_line)) {
        return false;
    }

    std::istringstream input(input_line);
    input >> std::ws;
    if (input.eof()) {
        value = default_value;
        return true;
    }

    if (!(input >> value)) {
        return false;
    }
    input >> std::ws;
    return input.eof();
}

} // namespace e_bts

#endif // E_BTS_CONSOLE_INPUT_UTILS_H
