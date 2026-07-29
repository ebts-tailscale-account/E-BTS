#ifndef E_BTS_GUI_WITTENSTEIN_BRIDGE_H
#define E_BTS_GUI_WITTENSTEIN_BRIDGE_H

#include "wittenstein_worker.h"

#include <QObject>
#include <QString>

namespace e_bts::gui {

// Same relay pattern as GuiBridge / ExportBridge (see gui_bridge.h for the
// full rationale): WittensteinWorker lives on its own QThread, and QML's
// Connections type refuses to bind to a target QObject living outside the
// QQmlEngine's thread. This object lives on the main thread and re-emits the
// worker's signals verbatim via QObject::connect, which does safely auto-queue
// cross-thread signals. QML listens to this (ftEvents), not ftWorker.
class WittensteinBridge : public QObject {
    Q_OBJECT

public:
    explicit WittensteinBridge(WittensteinWorker *worker, QObject *parent = nullptr) : QObject(parent) {
        connect(worker, &WittensteinWorker::sampleReady, this, &WittensteinBridge::sampleReady);
        connect(worker, &WittensteinWorker::connectionChanged, this, &WittensteinBridge::connectionChanged);
        connect(worker, &WittensteinWorker::logLine, this, &WittensteinBridge::logLine);
        connect(worker, &WittensteinWorker::recordingStateChanged, this, &WittensteinBridge::recordingStateChanged);
    }

signals:
    void sampleReady(double fx, double fy, double fz, double mx, double my, double mz, double temp);
    void connectionChanged(bool connected, QString info);
    void logLine(QString line);
    void recordingStateChanged(bool active);
};

} // namespace e_bts::gui

#endif // E_BTS_GUI_WITTENSTEIN_BRIDGE_H
