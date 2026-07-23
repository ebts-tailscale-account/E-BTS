#ifndef E_BTS_GUI_EXPORT_WORKER_H
#define E_BTS_GUI_EXPORT_WORKER_H

#include <QObject>
#include <QString>
#include <QVariantList>

namespace e_bts::gui {

// Runs raw_to_csv/raw_to_video conversions for files picked from the ribbon's
// Export dialog. Meant to live on its own QThread (moveToThread by the
// caller, same as CameraSessionWorker) since a conversion is a blocking loop
// over the whole file and must not stall the GUI thread -- see
// camera_session_worker.h for why that pattern exists. Independent of
// CameraSessionWorker: Camera::from_file() opens its own handle, so exporting
// does not contend with a live EVK1 session.
class ExportWorker : public QObject {
    Q_OBJECT

public:
    explicit ExportWorker(QObject *parent = nullptr) : QObject(parent) {}

public slots:
    // files: QVariantList of file:// url strings (see ExportDialog.qml for
    // why these are plain strings rather than the dialog's url values
    // directly). format: "csv" or "video". monochrome: only consulted for
    // "video" -- black & white rendering vs. the default polarity colors.
    void exportFiles(QVariantList files, QString format, bool monochrome = false);

signals:
    void logLine(QString line);
    void fileStarted(QString path);
    void fileFinished(QString path, bool success);
    void exportFinished(int succeeded, int failed);
};

} // namespace e_bts::gui

#endif // E_BTS_GUI_EXPORT_WORKER_H
