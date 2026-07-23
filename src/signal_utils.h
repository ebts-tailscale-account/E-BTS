#ifndef E_BTS_SIGNAL_UTILS_H
#define E_BTS_SIGNAL_UTILS_H

#include <csignal>

namespace e_bts {

inline volatile std::sig_atomic_t g_stop_requested = 0;

inline void request_stop(int) {
    g_stop_requested = 1;
}

inline void install_stop_signal_handlers() {
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
}

inline bool stop_requested() {
    return g_stop_requested != 0;
}

} // namespace e_bts

#endif // E_BTS_SIGNAL_UTILS_H
