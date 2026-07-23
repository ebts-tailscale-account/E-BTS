#ifndef E_BTS_GUI_LOG_STREAM_H
#define E_BTS_GUI_LOG_STREAM_H

#include <functional>
#include <ostream>
#include <streambuf>
#include <string>

namespace e_bts::gui {

// Adapts a std::ostream so every complete line written to it (via operator<<,
// same as std::cout/std::cerr) is handed to a callback instead of going to a
// terminal. Lets ExportWorker reuse camera_utils.h's convert_raw_to_csv /
// convert_raw_to_video exactly as the CLI tools call them -- out/err
// parameters -- while surfacing their progress as Qt signals.
class LineCallbackStreambuf : public std::streambuf {
public:
    explicit LineCallbackStreambuf(std::function<void(const std::string &)> on_line) :
        on_line_(std::move(on_line)) {}

protected:
    int_type overflow(int_type ch) override {
        if (ch == traits_type::eof()) {
            return traits_type::not_eof(ch);
        }
        if (ch == '\n') {
            flush_line();
        } else {
            buffer_.push_back(static_cast<char>(ch));
        }
        return ch;
    }

    int sync() override {
        flush_line();
        return 0;
    }

private:
    void flush_line() {
        if (!buffer_.empty()) {
            on_line_(buffer_);
            buffer_.clear();
        }
    }

    std::function<void(const std::string &)> on_line_;
    std::string buffer_;
};

// Pairs a LineCallbackStreambuf with the std::ostream that uses it, since the
// streambuf must outlive every write the stream makes.
class LineCallbackStream : public std::ostream {
public:
    explicit LineCallbackStream(std::function<void(const std::string &)> on_line) :
        std::ostream(&buf_), buf_(std::move(on_line)) {}

private:
    LineCallbackStreambuf buf_;
};

} // namespace e_bts::gui

#endif // E_BTS_GUI_LOG_STREAM_H
