// This target builds with strict -std=c++17 (CMAKE_CXX_EXTENSIONS OFF), which
// defines __STRICT_ANSI__ and otherwise hides the POSIX/BSD termios pieces we
// use (cfmakeraw, VMIN/VTIME, CLOCAL/CREAD). Requesting the default feature set
// before any system header is pulled in restores them. Must precede all includes.
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE 1
#endif

#include "wittenstein_worker.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <QTimer>

namespace e_bts::gui {

namespace {

constexpr int kPacketBytes   = 28;   // 7 x float32
constexpr int kPollIntervalMs = 5;
constexpr double kDisplayIntervalMs = 16.0; // ~60 Hz live graph

constexpr const char *kLogPrefix = "[E_BTS_GUI] [ft]";

inline bool temperature_plausible(float t) {
    return t > 0.0f && t < 80.0f;
}

} // namespace

WittensteinWorker::WittensteinWorker(QObject *parent) : QObject(parent) {
    std::cout << std::unitbuf; // flush the low-frequency connect/lock prints immediately
    last_emit_  = std::chrono::steady_clock::now();
    poll_timer_ = new QTimer(this);
    poll_timer_->setInterval(kPollIntervalMs);
    connect(poll_timer_, &QTimer::timeout, this, &WittensteinWorker::pollTick);
}

WittensteinWorker::~WittensteinWorker() {
    if (csv_.is_open()) {
        csv_.close();
    }
    closePort();
}

bool WittensteinWorker::openPort() {
    // The sensor is the only /dev/ttyACM* on this machine (the Franka is on the
    // network, the event camera is not a CDC ACM device), so scanning the first
    // few nodes and taking whichever opens is enough.
    static const char *candidates[] = {"/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2", "/dev/ttyACM3"};

    for (const char *path : candidates) {
        int fd = ::open(path, O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd < 0) {
            std::cout << kLogPrefix << " open(" << path << ") failed: " << std::strerror(errno) << '\n';
            continue;
        }

        termios tio{};
        if (::tcgetattr(fd, &tio) != 0) {
            std::cout << kLogPrefix << " tcgetattr(" << path << ") failed: " << std::strerror(errno) << '\n';
            ::close(fd);
            continue;
        }
        ::cfmakeraw(&tio);
        // 2,000,000 baud matches the manual's UART spec and the known-working
        // read_ft.py. On boxes that bridge USB-CDC to an internal UART the baud
        // is not a no-op, so match it rather than assume CDC ignores it.
        ::cfsetispeed(&tio, B2000000);
        ::cfsetospeed(&tio, B2000000);
        tio.c_cflag |= (CLOCAL | CREAD);
        tio.c_cc[VMIN]  = 0; // non-blocking: read() returns whatever is available
        tio.c_cc[VTIME] = 0;
        if (::tcsetattr(fd, TCSANOW, &tio) != 0) {
            std::cout << kLogPrefix << " tcsetattr(" << path << ") failed: " << std::strerror(errno) << '\n';
            ::close(fd);
            continue;
        }
        ::tcflush(fd, TCIFLUSH);

        // Assert DTR/RTS like pyserial does on open -- some VCP firmware only
        // starts streaming once the host raises these lines.
        int mstat = 0;
        if (::ioctl(fd, TIOCMGET, &mstat) == 0) {
            mstat |= (TIOCM_DTR | TIOCM_RTS);
            ::ioctl(fd, TIOCMSET, &mstat);
        }

        fd_        = fd;
        port_name_ = QString::fromLatin1(path);
        std::cout << kLogPrefix << " opened " << path << " @ 2000000 baud, DTR/RTS asserted.\n";
        return true;
    }
    std::cout << kLogPrefix << " no openable /dev/ttyACM0-3 found.\n";
    return false;
}

void WittensteinWorker::closePort() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    buffer_.clear();
    synced_ = false;
}

void WittensteinWorker::setActive(bool active) {
    // QML invokes this synchronously on the GUI thread, but the serial fd and
    // the poll QTimer must be created/started/driven on THIS worker's own
    // thread (a QTimer can't even be started from another thread). Hop over
    // with a queued call so applyActive() runs on the worker thread.
    QMetaObject::invokeMethod(this, "applyActive", Qt::QueuedConnection, Q_ARG(bool, active));
}

void WittensteinWorker::applyActive(bool active) {
    if (active) {
        if (fd_ >= 0) {
            return;
        }
        if (openPort()) {
            poll_timer_->start();
            emit connectionChanged(true, port_name_);
            emit logLine(QStringLiteral("F/T: connected on %1").arg(port_name_));
        } else {
            emit connectionChanged(false, QStringLiteral("no sensor found"));
            emit logLine(QStringLiteral(
                "F/T: no sensor on /dev/ttyACM0-3. Plugged in? Vendor GUI or read_ft.py still holding the port?"));
        }
    } else {
        poll_timer_->stop();
        closePort();
        emit connectionChanged(false, QStringLiteral("closed"));
    }
}

void WittensteinWorker::startRecording(const QString &raw_path) {
    if (recording_) {
        return;
    }

    // Share the camera .raw's base name + timestamp; write "<stem>_ft.csv" next
    // to it (the "_ft" suffix avoids colliding with the raw->csv export tool,
    // which produces "<stem>.csv").
    std::filesystem::path raw(raw_path.toStdString());
    std::filesystem::path csv_path = raw.parent_path() / (raw.stem().string() + "_ft.csv");

    csv_.open(csv_path);
    if (!csv_.is_open()) {
        emit logLine(QStringLiteral("F/T: could not open %1").arg(QString::fromStdString(csv_path.string())));
        return;
    }
    csv_ << "unix_time_s,Fx_N,Fy_N,Fz_N,Mx_mNm,My_mNm,Mz_mNm,Temp_C\n";
    recording_ = true;
    emit recordingStateChanged(true);

    if (fd_ < 0) {
        emit logLine(QStringLiteral("F/T: recording %1 but the sensor port is not open -- open the Force/Torque "
                                    "source or no data will be logged.")
                         .arg(QString::fromStdString(csv_path.filename().string())));
    } else {
        emit logLine(QStringLiteral("F/T: recording to %1").arg(QString::fromStdString(csv_path.filename().string())));
    }
}

void WittensteinWorker::stopRecording() {
    if (!recording_) {
        return;
    }
    csv_.flush();
    csv_.close();
    recording_ = false;
    emit recordingStateChanged(false);
    emit logLine(QStringLiteral("F/T: recording stopped."));
}

void WittensteinWorker::pollTick() {
    if (fd_ < 0) {
        return;
    }
    char chunk[8192];
    const ssize_t n = ::read(fd_, chunk, sizeof(chunk));
    if (n > 0) {
        buffer_.append(chunk, static_cast<int>(n));
    }
    parseBuffer();
}

int WittensteinWorker::findSync() const {
    // The temperature (7th float, bytes 24..27) sits in a narrow, roughly
    // constant range, so it's a reliable anchor. Require 3 packets in a row to
    // read plausibly before we trust an offset.
    for (int offset = 0; offset < kPacketBytes; ++offset) {
        bool ok = true;
        for (int k = 0; k < 3; ++k) {
            const int start = offset + k * kPacketBytes;
            if (start + kPacketBytes > buffer_.size()) {
                ok = false;
                break;
            }
            float temp;
            std::memcpy(&temp, buffer_.constData() + start + 24, sizeof(temp));
            if (!temperature_plausible(temp)) {
                ok = false;
                break;
            }
        }
        if (ok) {
            return offset;
        }
    }
    return -1;
}

void WittensteinWorker::parseBuffer() {
    if (!synced_) {
        const int offset = findSync();
        if (offset < 0) {
            // Keep the buffer from growing without bound while hunting.
            if (buffer_.size() > kPacketBytes * 40) {
                buffer_.remove(0, buffer_.size() - kPacketBytes * 4);
            }
            return;
        }
        if (offset > 0) {
            buffer_.remove(0, offset);
        }
        synced_ = true;
        std::cout << kLogPrefix << " locked onto packet stream (offset " << offset << ").\n";
    }

    while (buffer_.size() >= kPacketBytes) {
        float values[7];
        std::memcpy(values, buffer_.constData(), kPacketBytes);
        if (!temperature_plausible(values[6])) {
            // Slipped out of step -- drop one byte and realign.
            buffer_.remove(0, 1);
            continue;
        }
        buffer_.remove(0, kPacketBytes);
        handleSample(values);
    }
}

void WittensteinWorker::handleSample(const float *values) {
    if (recording_ && csv_.is_open()) {
        const double unix_s =
            std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        char line[192];
        std::snprintf(line, sizeof(line), "%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f\n", unix_s,
                      static_cast<double>(values[0]), static_cast<double>(values[1]),
                      static_cast<double>(values[2]), static_cast<double>(values[3]),
                      static_cast<double>(values[4]), static_cast<double>(values[5]),
                      static_cast<double>(values[6]));
        csv_ << line;
    }

    // Decimate the live-graph emissions; the CSV above already has every sample.
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double, std::milli>(now - last_emit_).count() >= kDisplayIntervalMs) {
        last_emit_ = now;
        emit sampleReady(values[0], values[1], values[2], values[3], values[4], values[5], values[6]);
    }
}

} // namespace e_bts::gui
