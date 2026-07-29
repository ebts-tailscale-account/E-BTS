#ifndef E_BTS_GUI_WITTENSTEIN_WORKER_H
#define E_BTS_GUI_WITTENSTEIN_WORKER_H

#include <chrono>
#include <fstream>

#include <QByteArray>
#include <QObject>
#include <QString>

class QTimer;

namespace e_bts::gui {

// Reads the Resense / Wittenstein HEX21 6-axis force/torque sensor over its
// USB CDC virtual COM port (STMicroelectronics 0483:5740, /dev/ttyACM*). With
// electronics DIP switch 6 = ON the box streams *calculated* F/T as 28-byte
// packets of seven little-endian float32s: Fx Fy Fz (N), Mx My Mz (mNm),
// Temp (C). There are no framing bytes, so we lock onto packet boundaries by
// sliding until the temperature reads plausible, and re-sync the same way if a
// byte ever slips.
//
// Same worker-on-its-own-QThread + main-thread bridge pattern as
// CameraSessionWorker (see gui/gui_bridge.h for why the relay exists). A
// QTimer drains the port every few ms rather than blocking, so the thread's
// event loop still processes queued slots -- crucially startRecording() /
// stopRecording(), which the camera recorder fires (queued, cross-thread) so
// the F/T CSV starts and stops in lockstep with the camera .raw.
class WittensteinWorker : public QObject {
    Q_OBJECT

public:
    explicit WittensteinWorker(QObject *parent = nullptr);
    ~WittensteinWorker() override;

public slots:
    // Open/close the serial port and start/stop streaming. Toggled by the
    // Force/Torque entry in the ribbon's Add Source menu.
    void setActive(bool active);

    // Full-rate CSV logging. Driven by the camera recorder's start/stop so the
    // CSV shares the .raw's base name + timestamp: given
    // ".../name_YYYYMMDD_HHMMSS.raw" it writes ".../name_YYYYMMDD_HHMMSS_ft.csv"
    // right next to it. Every sample is stamped with UNIX wall-clock time (the
    // cross-stream alignment anchor).
    void startRecording(const QString &raw_path);
    void stopRecording();

signals:
    // Decimated to ~60 Hz for the live graph; the CSV is written at full rate.
    void sampleReady(double fx, double fy, double fz, double mx, double my, double mz, double temp);
    void connectionChanged(bool connected, QString info);
    void logLine(QString line);
    void recordingStateChanged(bool active);

private slots:
    // Runs the actual open/close on the worker thread (setActive() only
    // marshals here) so the serial fd and poll QTimer live on and are driven
    // from this thread.
    void applyActive(bool active);
    void pollTick();

private:
    bool openPort();
    void closePort();
    int findSync() const;   // byte offset of the first aligned packet, or -1
    void parseBuffer();
    void handleSample(const float *values);

    int fd_ = -1;
    QString port_name_;
    QByteArray buffer_;
    bool synced_ = false;

    QTimer *poll_timer_ = nullptr;

    std::ofstream csv_;
    bool recording_ = false;

    std::chrono::steady_clock::time_point last_emit_;
};

} // namespace e_bts::gui

#endif // E_BTS_GUI_WITTENSTEIN_WORKER_H
