#include "export_worker.h"

#include <iostream>

#include "log_stream.h"

#include "../camera_feed_utils.h"
#include "../camera_utils.h"
#include "../path_utils.h"

#include <QUrl>

namespace e_bts::gui {

void ExportWorker::exportFiles(QVariantList files, QString format, bool monochrome) {
    const bool as_video = format == QStringLiteral("video");
    const e_bts::CameraFeedMode feed_mode =
        monochrome ? e_bts::CameraFeedMode::Monochrome : e_bts::CameraFeedMode::Default;

    std::cerr << "[E_BTS_GUI] [export] Exporting " << files.size() << " file(s) as " << format.toStdString()
              << (as_video && monochrome ? " (black & white)" : "") << ".\n";

    int succeeded = 0;
    int failed    = 0;

    for (const QVariant &file : files) {
        const QString local_path = file.toUrl().toLocalFile();
        const std::filesystem::path input_path = local_path.toStdString();

        emit fileStarted(local_path);

        LineCallbackStream out([this](const std::string &line) {
            std::cerr << "[E_BTS_GUI] [export] " << line << '\n';
            emit logLine(QString::fromStdString(line));
        });
        LineCallbackStream err([this](const std::string &line) {
            std::cerr << "[E_BTS_GUI] [export] " << line << '\n';
            emit logLine(QString::fromStdString(line));
        });

        bool success = false;
        if (as_video) {
            success = e_bts::convert_raw_to_video(input_path, e_bts::default_video_path(input_path, monochrome),
                                                   /*accumulation_time_us=*/10'000, feed_mode, out, err)
                          .success;
        } else {
            success = e_bts::convert_raw_to_csv(input_path, e_bts::default_csv_path(input_path), nullptr, out, err)
                          .success;
        }

        if (success) {
            ++succeeded;
        } else {
            ++failed;
        }
        emit fileFinished(local_path, success);
    }

    emit exportFinished(succeeded, failed);
}

} // namespace e_bts::gui
