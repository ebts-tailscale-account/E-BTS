#ifndef E_BTS_GUI_EXPORT_BRIDGE_H
#define E_BTS_GUI_EXPORT_BRIDGE_H

#include "export_worker.h"

#include <QObject>
#include <QString>

namespace e_bts::gui {

// Same relay pattern as GuiBridge (see gui_bridge.h for the full rationale):
// ExportWorker lives on its own QThread, and QML's Connections type refuses
// to bind to a target QObject living outside the QQmlEngine's thread. This
// object lives on the main thread and re-emits ExportWorker's signals
// verbatim via QObject::connect, which does support cross-thread signals.
class ExportBridge : public QObject {
    Q_OBJECT

public:
    explicit ExportBridge(ExportWorker *worker, QObject *parent = nullptr) : QObject(parent) {
        connect(worker, &ExportWorker::logLine, this, &ExportBridge::logLine);
        connect(worker, &ExportWorker::fileStarted, this, &ExportBridge::fileStarted);
        connect(worker, &ExportWorker::fileFinished, this, &ExportBridge::fileFinished);
        connect(worker, &ExportWorker::exportFinished, this, &ExportBridge::exportFinished);
    }

signals:
    void logLine(QString line);
    void fileStarted(QString path);
    void fileFinished(QString path, bool success);
    void exportFinished(int succeeded, int failed);
};

} // namespace e_bts::gui

#endif // E_BTS_GUI_EXPORT_BRIDGE_H
