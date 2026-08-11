// E-BTS Metavision bias tuner — standalone one-off calibration/testing GUI.
//
// Opens the first available EVK1, shows the live event view + event-rate stats,
// and one slider per hardware bias (auto-discovered via the HAL I_LL_Biases
// facility -- e.g. bias_diff_on/off contrast, bias_hpf HPF, bias_fo LPF,
// bias_refr). Changes apply on slider *release* (each set/get is a blocking USB
// round-trip, so applying continuously would freeze the UI); out-of-range
// values snap back to what the sensor accepted. The accumulation-time spinbox
// retunes the live view.
//
// "Save biases…" writes the Metavision .bias format (`<value> % <bias_name>`
// per line, loadable by Studio/HAL); "Load biases…" reads that format back and
// pushes it to the sensor (the device does NOT retain biases between sessions,
// so loading is how you resume a previous tuning session).
//
// HARDWARE ROI (I_ROI): drag a rectangle on the live view, or type x/y/w/h, to
// mask the sensor readout itself -- events outside the box are never generated,
// so the event rate drops at the source. This is the hardware ROI, distinct
// from any software cropping downstream.
//
// SINGLE-OWNER CAMERA: close E_BTS_GUI and the vendor GUI first. Quit via the
// window's X (not Ctrl-C) so the camera is released cleanly.
//
// Separate mini-project. Build:
//     cmake -S calibration/bias_tuner -B calibration/bias_tuner/build
//     cmake --build calibration/bias_tuner/build -j
// Run:
//     ./calibration/bias_tuner/build/bias_tuner

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <QApplication>
#include <QCheckBox>
#include <QDateTime>
#include <QDir>
#include <QEvent>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPixmap>
#include <QPushButton>
#include <QRegExp>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSlider>
#include <QSpinBox>
#include <QStringList>
#include <QTextStream>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <metavision/hal/device/device.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/facilities/i_roi.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <metavision/sdk/driver/camera.h>

namespace {

struct SharedFrame {
    std::mutex mutex;
    QImage image;
    bool has_new = false;
};

struct EventStats {
    std::atomic<std::uint64_t> on{0};
    std::atomic<std::uint64_t> off{0};
};

// Rectangles drawn over the live frame. Written by the GUI thread (mouse drag /
// spinboxes), read by the frame-generator thread -> guarded.
struct RoiOverlay {
    std::mutex mutex;
    cv::Rect roi;             // last applied ROI, in sensor pixels
    bool roi_enabled = false; // drawn green when enabled
    cv::Rect drag;            // in-progress drag, drawn yellow
    bool dragging = false;
};

QString human_rate(double v) {
    if (v >= 1e6) {
        return QString::number(v / 1e6, 'f', 2) + " M";
    }
    if (v >= 1e3) {
        return QString::number(v / 1e3, 'f', 1) + " k";
    }
    return QString::number(v, 'f', 0);
}

// The live view is an aspect-preserved, centre-aligned pixmap inside the label,
// so widget coords must be un-letterboxed before they mean sensor pixels.
bool widget_to_sensor(const QPoint &pos, const QSize &view_size, int width, int height, QPoint &out) {
    const double scale =
        std::min(static_cast<double>(view_size.width()) / width, static_cast<double>(view_size.height()) / height);
    if (scale <= 0.0) {
        return false;
    }
    const double offset_x = (view_size.width() - width * scale) / 2.0;
    const double offset_y = (view_size.height() - height * scale) / 2.0;
    const int x = static_cast<int>((pos.x() - offset_x) / scale);
    const int y = static_cast<int>((pos.y() - offset_y) / scale);
    out.setX(std::max(0, std::min(width - 1, x)));
    out.setY(std::max(0, std::min(height - 1, y)));
    return true;
}

// Drag-to-select on the live view. Plain QObject event filter -> no Q_OBJECT /
// moc needed for this one-off tool.
class ViewDragFilter : public QObject {
public:
    ViewDragFilter(QLabel *view, int width, int height, std::shared_ptr<RoiOverlay> overlay,
                   std::function<void(cv::Rect)> on_selected) :
        view_(view),
        width_(width),
        height_(height),
        overlay_(std::move(overlay)),
        on_selected_(std::move(on_selected)) {}

protected:
    bool eventFilter(QObject *object, QEvent *event) override {
        if (object != view_) {
            return QObject::eventFilter(object, event);
        }
        if (event->type() == QEvent::MouseButtonPress) {
            auto *mouse = static_cast<QMouseEvent *>(event);
            if (mouse->button() != Qt::LeftButton) {
                return false;
            }
            if (!widget_to_sensor(mouse->pos(), view_->size(), width_, height_, anchor_)) {
                return false;
            }
            active_ = true;
            publish(anchor_);
            return true;
        }
        if (event->type() == QEvent::MouseMove && active_) {
            QPoint current;
            if (widget_to_sensor(static_cast<QMouseEvent *>(event)->pos(), view_->size(), width_, height_, current)) {
                publish(current);
            }
            return true;
        }
        if (event->type() == QEvent::MouseButtonRelease && active_) {
            active_ = false;
            QPoint current;
            if (!widget_to_sensor(static_cast<QMouseEvent *>(event)->pos(), view_->size(), width_, height_, current)) {
                return false;
            }
            const cv::Rect rect = make_rect(anchor_, current);
            {
                std::lock_guard<std::mutex> lock(overlay_->mutex);
                overlay_->dragging = false;
            }
            // A stray click (or a sliver) would blind the sensor -- ignore it.
            if (rect.width >= 4 && rect.height >= 4) {
                on_selected_(rect);
            }
            return true;
        }
        return QObject::eventFilter(object, event);
    }

private:
    cv::Rect make_rect(const QPoint &a, const QPoint &b) const {
        const int x = std::min(a.x(), b.x());
        const int y = std::min(a.y(), b.y());
        return cv::Rect(x, y, std::abs(a.x() - b.x()) + 1, std::abs(a.y() - b.y()) + 1);
    }

    void publish(const QPoint &current) {
        std::lock_guard<std::mutex> lock(overlay_->mutex);
        overlay_->drag     = make_rect(anchor_, current);
        overlay_->dragging = true;
    }

    QLabel *view_;
    int width_;
    int height_;
    std::shared_ptr<RoiOverlay> overlay_;
    std::function<void(cv::Rect)> on_selected_;
    QPoint anchor_;
    bool active_ = false;
};

} // namespace

int main(int argc, char **argv) {
    QApplication app(argc, argv);

    // ---- open the camera (single-owner) ----
    std::optional<Metavision::Camera> camera;
    try {
        camera.emplace(Metavision::Camera::from_first_available());
    } catch (const std::exception &error) {
        QMessageBox::critical(nullptr, "Bias Tuner",
                              QString("Could not open the EVK1:\n%1\n\nIs it plugged in and free? "
                                      "Close E_BTS_GUI and the vendor GUI first (one handle only).")
                                  .arg(error.what()));
        return 1;
    }
    const int width  = camera->geometry().width();
    const int height = camera->geometry().height();

    Metavision::I_LL_Biases *biases = nullptr;
    try {
        biases = camera->get_device().get_facility<Metavision::I_LL_Biases>();
    } catch (const std::exception &) {
        biases = nullptr;
    }
    if (biases == nullptr) {
        QMessageBox::critical(nullptr, "Bias Tuner", "This camera exposes no I_LL_Biases facility.");
        return 1;
    }
    const std::map<std::string, int> initial_biases = biases->get_all_biases();

    // ROI is optional -- absence disables the section rather than the tool.
    Metavision::I_ROI *roi_facility = nullptr;
    try {
        roi_facility = camera->get_device().get_facility<Metavision::I_ROI>();
    } catch (const std::exception &) {
        roi_facility = nullptr;
    }

    // shared state for the live view + stats + accumulation time
    auto shared   = std::make_shared<SharedFrame>();
    auto stats    = std::make_shared<EventStats>();
    auto accum_us = std::make_shared<std::atomic<int>>(10'000);
    auto overlay  = std::make_shared<RoiOverlay>();

    Metavision::CDFrameGenerator frame_gen(width, height);
    frame_gen.set_display_accumulation_time_us(accum_us->load());

    // ---- window ----
    QWidget window;
    window.setWindowTitle("E-BTS — Metavision bias tuner (calibration)");
    window.resize(1100, 1010);
    window.setStyleSheet("QLabel { font-size: 14px; }"
                         "QSlider { min-height: 30px; }"
                         "QSpinBox { min-height: 28px; font-size: 14px; }"
                         "QPushButton { font-size: 15px; min-height: 40px; }");
    auto *root = new QVBoxLayout(&window);

    auto *hint = new QLabel(QString("EVK1 %1x%2 — drag to tune (applies on release); out-of-range snaps back. "
                                    "Drag on the view to set the hardware ROI.")
                                .arg(width)
                                .arg(height));
    hint->setStyleSheet("color:#888;");
    root->addWidget(hint);

    // live event view (takes the extra vertical space -> big, aspect-preserved)
    auto *view = new QLabel("waiting for events…");
    view->setAlignment(Qt::AlignCenter);
    view->setMinimumHeight(420);
    view->setStyleSheet("background:#111; color:#888;");
    view->setMouseTracking(true);
    root->addWidget(view, 1);

    // ---- statistics section ----
    auto *stats_label = new QLabel("events/s — ON: —  OFF: —");
    stats_label->setStyleSheet("font-family: monospace; color:#1B5E20;");
    root->addWidget(stats_label);

    auto *status = new QLabel;
    status->setStyleSheet("color:#4CAF50;");

    // ---- accumulation-time control ----
    {
        auto *acc_row   = new QHBoxLayout;
        auto *acc_label = new QLabel("Accumulation time (µs):");
        auto *acc_spin  = new QSpinBox;
        acc_spin->setRange(100, 1'000'000);
        acc_spin->setSingleStep(1'000);
        acc_spin->setValue(accum_us->load());
        acc_spin->setMinimumWidth(130);
        QObject::connect(acc_spin, QOverload<int>::of(&QSpinBox::valueChanged),
                         [&frame_gen, accum_us](int v) {
                             accum_us->store(v);
                             frame_gen.set_display_accumulation_time_us(v);
                         });
        acc_row->addWidget(acc_label);
        acc_row->addWidget(acc_spin);
        acc_row->addStretch(1);
        root->addLayout(acc_row);
    }

    // ---- hardware ROI section ----
    // Filled in by the section below so Save/Load (declared later, outside this
    // scope) can persist the ROI geometry alongside the biases. nullopt = full
    // frame.
    auto roi_get = std::make_shared<std::function<std::optional<cv::Rect>()>>();
    auto roi_put = std::make_shared<std::function<void(const std::optional<cv::Rect> &)>>();

    auto *roi_group = new QGroupBox("Hardware ROI (I_ROI — masks the sensor readout)");
    {
        auto *roi_row   = new QHBoxLayout(roi_group);
        auto *roi_check = new QCheckBox("Enable");
        auto *x_spin    = new QSpinBox;
        auto *y_spin    = new QSpinBox;
        auto *w_spin    = new QSpinBox;
        auto *h_spin    = new QSpinBox;
        x_spin->setRange(0, width - 1);
        y_spin->setRange(0, height - 1);
        w_spin->setRange(1, width);
        h_spin->setRange(1, height);
        x_spin->setValue(0);
        y_spin->setValue(0);
        w_spin->setValue(width);
        h_spin->setValue(height);
        for (QSpinBox *spin : {x_spin, y_spin, w_spin, h_spin}) {
            spin->setMinimumWidth(85);
        }
        auto *apply_button = new QPushButton("Apply ROI");
        auto *full_button  = new QPushButton("Full frame");

        // Clamp so the box always lies inside the sensor, then program it. The
        // cols/rows binary-map overload is pure-virtual in HAL 3.1.2 and a
        // rectangle is exactly "these columns AND these rows".
        auto apply_roi = [roi_facility, width, height, overlay, status, roi_check, x_spin, y_spin, w_spin,
                          h_spin]() {
            if (roi_facility == nullptr) {
                return;
            }
            const int x = std::min(x_spin->value(), width - 1);
            const int y = std::min(y_spin->value(), height - 1);
            const int w = std::min(w_spin->value(), width - x);
            const int h = std::min(h_spin->value(), height - y);
            {
                QSignalBlocker block_w(w_spin);
                QSignalBlocker block_h(h_spin);
                w_spin->setValue(w);
                h_spin->setValue(h);
            }
            const bool enabled  = roi_check->isChecked();
            const bool whole    = (x == 0 && y == 0 && w == width && h == height);
            std::vector<bool> cols(static_cast<std::size_t>(width), !enabled || whole);
            std::vector<bool> rows(static_cast<std::size_t>(height), !enabled || whole);
            if (enabled && !whole) {
                for (int col = x; col < x + w; ++col) {
                    cols[static_cast<std::size_t>(col)] = true;
                }
                for (int row = y; row < y + h; ++row) {
                    rows[static_cast<std::size_t>(row)] = true;
                }
            }
            bool ok = false;
            try {
                ok = roi_facility->set_ROIs(cols, rows, enabled);
            } catch (const std::exception &error) {
                status->setText(QString("ROI failed: %1").arg(error.what()));
                return;
            }
            {
                std::lock_guard<std::mutex> lock(overlay->mutex);
                overlay->roi         = cv::Rect(x, y, w, h);
                overlay->roi_enabled = enabled;
            }
            if (!ok) {
                status->setText("ROI rejected by the sensor (check x/y/w/h).");
                return;
            }
            status->setText(enabled ? QString("ROI on — x=%1 y=%2 w=%3 h=%4 (%5 px, %6% of frame)")
                                          .arg(x)
                                          .arg(y)
                                          .arg(w)
                                          .arg(h)
                                          .arg(w * h)
                                          .arg(100.0 * w * h / (width * height), 0, 'f', 1)
                                    : QString("ROI off (full %1x%2 frame)").arg(width).arg(height));
        };

        *roi_get = [roi_check, x_spin, y_spin, w_spin, h_spin]() -> std::optional<cv::Rect> {
            if (!roi_check->isChecked()) {
                return std::nullopt;
            }
            return cv::Rect(x_spin->value(), y_spin->value(), w_spin->value(), h_spin->value());
        };
        *roi_put = [apply_roi, roi_check, x_spin, y_spin, w_spin, h_spin, width,
                    height](const std::optional<cv::Rect> &rect) {
            QSignalBlocker block_check(roi_check);
            x_spin->setValue(rect ? rect->x : 0);
            y_spin->setValue(rect ? rect->y : 0);
            w_spin->setValue(rect ? rect->width : width);
            h_spin->setValue(rect ? rect->height : height);
            roi_check->setChecked(rect.has_value());
            apply_roi();
        };

        QObject::connect(apply_button, &QPushButton::clicked, apply_roi);
        QObject::connect(roi_check, &QCheckBox::toggled, [apply_roi](bool) { apply_roi(); });
        QObject::connect(full_button, &QPushButton::clicked,
                         [apply_roi, roi_check, x_spin, y_spin, w_spin, h_spin, width, height]() {
                             QSignalBlocker block_check(roi_check);
                             x_spin->setValue(0);
                             y_spin->setValue(0);
                             w_spin->setValue(width);
                             h_spin->setValue(height);
                             roi_check->setChecked(false);
                             apply_roi();
                         });

        roi_row->addWidget(roi_check);
        for (const auto &[text, spin] : std::initializer_list<std::pair<const char *, QSpinBox *>>{
                 {"x", x_spin}, {"y", y_spin}, {"w", w_spin}, {"h", h_spin}}) {
            roi_row->addWidget(new QLabel(text));
            roi_row->addWidget(spin);
        }
        roi_row->addWidget(apply_button);
        roi_row->addWidget(full_button);
        roi_row->addStretch(1);

        if (roi_facility == nullptr) {
            roi_group->setEnabled(false);
            roi_group->setTitle("Hardware ROI — unavailable (no I_ROI facility on this device)");
        } else {
            // Drag on the live view -> fill the spinboxes, enable, apply.
            auto *drag_filter = new ViewDragFilter(
                view, width, height, overlay,
                [apply_roi, roi_check, x_spin, y_spin, w_spin, h_spin](cv::Rect rect) {
                    QSignalBlocker block_check(roi_check);
                    x_spin->setValue(rect.x);
                    y_spin->setValue(rect.y);
                    w_spin->setValue(rect.width);
                    h_spin->setValue(rect.height);
                    roi_check->setChecked(true);
                    apply_roi();
                });
            view->installEventFilter(drag_filter);
        }
    }
    root->addWidget(roi_group);

    // ---- bias sliders ----
    auto *scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setMinimumHeight(300);
    scroll->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    auto *list        = new QWidget;
    auto *list_layout = new QVBoxLayout(list);
    list_layout->setSpacing(10);
    scroll->setWidget(list);
    root->addWidget(scroll, 0);

    auto value_spins  = std::make_shared<std::map<std::string, QSpinBox *>>();
    auto bias_applies = std::make_shared<std::map<std::string, std::function<void(int)>>>();

    for (const auto &[name, value] : initial_biases) {
        auto *row    = new QHBoxLayout;
        auto *label  = new QLabel(QString::fromStdString(name));
        auto *slider = new QSlider(Qt::Horizontal);
        auto *spin   = new QSpinBox;
        label->setMinimumWidth(170);
        spin->setMinimumWidth(90);

        // No range API on I_LL_Biases -> generous bounds; overshoot snaps back on
        // release (set/get read-back), and warnings are silenced via setLogLevel.
        const int max_value = std::max(1800, value * 2 + 100);
        slider->setRange(0, max_value);
        spin->setRange(0, max_value);
        slider->setValue(value);
        spin->setValue(value);

        auto apply = [biases, name, slider, spin](int requested) {
            biases->set(name, requested);
            const int applied = biases->get(name);
            QSignalBlocker block_slider(slider);
            QSignalBlocker block_spin(spin);
            slider->setValue(applied);
            spin->setValue(applied);
        };
        QObject::connect(slider, &QSlider::valueChanged, [spin](int v) {
            QSignalBlocker block_spin(spin);
            spin->setValue(v);
        });
        QObject::connect(slider, &QSlider::sliderReleased, [apply, slider]() { apply(slider->value()); });
        QObject::connect(spin, QOverload<int>::of(&QSpinBox::valueChanged), [apply](int v) { apply(v); });

        (*value_spins)[name]  = spin;
        (*bias_applies)[name] = apply;
        row->addWidget(label);
        row->addWidget(slider, 1);
        row->addWidget(spin);
        list_layout->addLayout(row);
    }
    list_layout->addStretch(1);

    root->addWidget(status);

    // ---- load / save ----
    {
        auto *button_row = new QHBoxLayout;
        auto *load       = new QPushButton("Load biases…");
        auto *save       = new QPushButton("Save biases…");
        button_row->addWidget(load);
        button_row->addWidget(save);
        root->addLayout(button_row);

        QObject::connect(load, &QPushButton::clicked, [&window, bias_applies, value_spins, status, roi_put]() {
            const QString path =
                QFileDialog::getOpenFileName(&window, "Load biases", QString(), "Bias files (*.bias);;All files (*)");
            if (path.isEmpty()) {
                return;
            }
            QFile file(path);
            if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
                status->setText("Load failed: " + path);
                return;
            }
            // Metavision .bias format: "<value> % <bias_name>" per line. Padding
            // varies between writers ("221  % bias_diff_off"), so split loosely.
            QTextStream in(&file);
            QStringList unknown;
            QStringList rejected;
            int applied_count = 0;
            while (!in.atEnd()) {
                const QString line = in.readLine().trimmed();
                if (line.isEmpty() || line.startsWith('#')) {
                    continue;
                }
                const int separator = line.indexOf('%');
                if (separator < 0) {
                    continue;
                }
                bool numeric      = false;
                const int value   = line.left(separator).trimmed().toInt(&numeric);
                const QString key = line.mid(separator + 1).trimmed();
                if (!numeric || key.isEmpty()) {
                    continue;
                }
                const auto entry = bias_applies->find(key.toStdString());
                if (entry == bias_applies->end()) {
                    unknown.append(key);
                    continue;
                }
                entry->second(value); // set + read-back; the spin shows what stuck
                if ((*value_spins)[key.toStdString()]->value() != value) {
                    rejected.append(QString("%1 (asked %2, got %3)")
                                        .arg(key)
                                        .arg(value)
                                        .arg((*value_spins)[key.toStdString()]->value()));
                }
                ++applied_count;
            }
            file.close();

            // Restore the companion ROI, if this .bias has one.
            const QFileInfo info(path);
            const QString roi_path = info.dir().filePath(info.completeBaseName() + ".roi");
            QString roi_message;
            QFile roi_file(roi_path);
            if (roi_file.exists() && roi_file.open(QIODevice::ReadOnly | QIODevice::Text)) {
                QTextStream roi_in(&roi_file);
                std::optional<cv::Rect> rect;
                while (!roi_in.atEnd()) {
                    const QString roi_line = roi_in.readLine().trimmed();
                    if (roi_line.isEmpty() || roi_line.startsWith('#')) {
                        continue;
                    }
                    const QStringList fields = roi_line.split(QRegExp("\\s+"), QString::SkipEmptyParts);
                    if (!fields.isEmpty() &&
                        (fields[0] == "disabled" || fields[0] == "off" || fields[0] == "full")) {
                        break; // explicit full-frame
                    }
                    if (fields.size() >= 4) {
                        bool ok_x = false, ok_y = false, ok_w = false, ok_h = false;
                        const int x = fields[0].toInt(&ok_x);
                        const int y = fields[1].toInt(&ok_y);
                        const int w = fields[2].toInt(&ok_w);
                        const int h = fields[3].toInt(&ok_h);
                        if (ok_x && ok_y && ok_w && ok_h) {
                            rect = cv::Rect(x, y, w, h);
                        }
                    }
                    break;
                }
                roi_file.close();
                if (*roi_put) {
                    (*roi_put)(rect);
                }
                roi_message = rect ? QString(" — ROI %1 %2 %3 %4")
                                         .arg(rect->x)
                                         .arg(rect->y)
                                         .arg(rect->width)
                                         .arg(rect->height)
                                   : QString(" — ROI off");
            }

            QString message = QString("Loaded %1 bias%2 from %3")
                                  .arg(applied_count)
                                  .arg(applied_count == 1 ? "" : "es")
                                  .arg(QFileInfo(path).fileName());
            message += roi_message;
            if (!rejected.isEmpty()) {
                message += " — clamped: " + rejected.join(", ");
            }
            if (!unknown.isEmpty()) {
                message += " — unknown: " + unknown.join(", ");
            }
            status->setText(message);
        });

        QObject::connect(save, &QPushButton::clicked, [&window, value_spins, status, roi_get, width, height]() {
            const QString suggested =
                QString("biases_%1.bias").arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
            const QString path = QFileDialog::getSaveFileName(&window, "Save biases", suggested, "Bias files (*.bias)");
            if (path.isEmpty()) {
                return;
            }
            QFile file(path);
            if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
                status->setText("Save failed: " + path);
                return;
            }
            // Metavision .bias format: "<value> % <bias_name>" per line.
            QTextStream out(&file);
            for (const auto &[name, spin] : *value_spins) {
                out << spin->value() << " % " << QString::fromStdString(name) << "\n";
            }
            file.close();

            // The .bias format has no ROI field (and the .raw header only ever
            // reports full sensor geometry), so the ROI goes in a companion file
            // that E_BTS_GUI reads back alongside the biases. Written even when
            // the ROI is off, so "off" is recorded rather than merely absent.
            const QFileInfo info(path);
            const QString roi_path        = info.dir().filePath(info.completeBaseName() + ".roi");
            const std::optional<cv::Rect> roi = (*roi_get) ? (*roi_get)() : std::nullopt;
            QFile roi_file(roi_path);
            if (!roi_file.open(QIODevice::WriteOnly | QIODevice::Text)) {
                status->setText("Saved " + path + " — but ROI write failed: " + roi_path);
                return;
            }
            QTextStream roi_out(&roi_file);
            // ASCII only: QTextStream encodes with the locale codec, which
            // mangles non-ASCII (an em-dash here came out as "â" under LANG=C).
            roi_out << "# E-BTS hardware ROI (I_ROI) - x y width height, in sensor pixels.\n";
            roi_out << "# Sensor geometry " << width << "x" << height
                    << "; event coordinates in the .raw stay absolute (NOT re-based to the ROI origin).\n";
            if (roi) {
                roi_out << roi->x << ' ' << roi->y << ' ' << roi->width << ' ' << roi->height << '\n';
            } else {
                roi_out << "disabled\n";
            }
            roi_file.close();
            status->setText(QString("Saved %1 + %2 (ROI %3)")
                                .arg(QFileInfo(path).fileName(), QFileInfo(roi_path).fileName(),
                                     roi ? QString("%1 %2 %3 %4")
                                               .arg(roi->x)
                                               .arg(roi->y)
                                               .arg(roi->width)
                                               .arg(roi->height)
                                         : QString("off")));
        });
    }

    // ---- live event view + stats pipeline ----
    camera->cd().add_callback([&frame_gen, stats](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::uint64_t on_count = 0;
        for (const Metavision::EventCD *it = begin; it != end; ++it) {
            on_count += (it->p != 0) ? 1u : 0u;
        }
        const std::uint64_t total = static_cast<std::uint64_t>(end - begin);
        stats->on.fetch_add(on_count, std::memory_order_relaxed);
        stats->off.fetch_add(total - on_count, std::memory_order_relaxed);
        frame_gen.add_events(begin, end);
    });
    frame_gen.start(30, [shared, overlay](Metavision::timestamp, cv::Mat &frame) {
        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        {
            std::lock_guard<std::mutex> lock(overlay->mutex);
            if (overlay->roi_enabled) {
                cv::rectangle(rgb, overlay->roi, cv::Scalar(0, 255, 0), 1);
            }
            if (overlay->dragging) {
                cv::rectangle(rgb, overlay->drag, cv::Scalar(255, 220, 0), 1);
            }
        }
        const QImage image(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
        std::lock_guard<std::mutex> lock(shared->mutex);
        shared->image   = image.copy();
        shared->has_new = true;
    });

    QTimer view_timer;
    QObject::connect(&view_timer, &QTimer::timeout, [view, shared]() {
        QImage image;
        {
            std::lock_guard<std::mutex> lock(shared->mutex);
            if (!shared->has_new) {
                return;
            }
            image           = shared->image;
            shared->has_new = false;
        }
        view->setPixmap(QPixmap::fromImage(image).scaled(view->size(), Qt::KeepAspectRatio,
                                                         Qt::SmoothTransformation));
    });
    view_timer.start(33);

    // once-a-second event-rate statistics
    QTimer stats_timer;
    QObject::connect(&stats_timer, &QTimer::timeout,
                     [stats_label, stats, accum_us, last_on = std::uint64_t(0),
                      last_off = std::uint64_t(0)]() mutable {
                         const std::uint64_t on  = stats->on.load(std::memory_order_relaxed);
                         const std::uint64_t off = stats->off.load(std::memory_order_relaxed);
                         const double on_ps  = static_cast<double>(on - last_on);   // events in the last 1 s
                         const double off_ps = static_cast<double>(off - last_off);
                         last_on  = on;
                         last_off = off;
                         const double window_s = accum_us->load() / 1e6;
                         stats_label->setText(
                             QString("events/s — ON %1  OFF %2  total %3      |      per %4 µs window — ON %5  OFF %6")
                                 .arg(human_rate(on_ps), human_rate(off_ps), human_rate(on_ps + off_ps))
                                 .arg(accum_us->load())
                                 .arg(human_rate(on_ps * window_s), human_rate(off_ps * window_s)));
                     });
    stats_timer.start(1'000);

    camera->start();
    Metavision::setLogLevel(Metavision::LogLevel::Error); // hush per-drag [HAL][WARNING] bias-range spam

    QObject::connect(&app, &QApplication::aboutToQuit, [&]() {
        view_timer.stop();
        stats_timer.stop();
        frame_gen.stop();
        if (camera) {
            camera->stop();
        }
    });

    window.show();
    return app.exec();
}
