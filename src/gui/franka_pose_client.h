#ifndef E_BTS_GUI_FRANKA_POSE_CLIENT_H
#define E_BTS_GUI_FRANKA_POSE_CLIENT_H

#include <QByteArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QObject>
#include <QString>
#include <QTimer>
#include <QUrl>

namespace e_bts::gui {

// Polls franka/pose_server.py on tactile for the live end-effector XY, so the
// camera's contact estimate can be checked against the robot's own answer while
// the arm is moving.
//
// THE TWO NUMBERS ARE IN THE SAME FRAME ON PURPOSE. calibration/pixel_to_mm.json
// was fitted from franka/calib_raster.py against robot-base millimetres, so the
// difference between them is a subtraction and nothing else -- no rotation, no
// handedness convention, nothing that could be silently wrong by a sign.
//
// ⚠ A STALE POSE IS NOT A POSE. franka_states stops publishing entirely when the
// user-stop is pressed or a reflex latches. The server reports the age of its last
// message and this client treats anything older than kStaleAfterSeconds as no
// pose at all, because a frozen "truth" would quietly turn a moving estimate into
// a growing, meaningless error.
//
// Polling, not streaming: the whole point is a human-readable readout at a few Hz
// next to a 20 Hz estimate, and a poll needs no reconnect logic, no backpressure
// and no second thread. It costs one small HTTP round trip per tick on a tailnet
// that already carries the force bridge.
class FrankaPoseClient : public QObject {
    Q_OBJECT

public:
    explicit FrankaPoseClient(QObject *parent = nullptr) : QObject(parent) {
        timer_ = new QTimer(this);
        timer_->setInterval(kPollIntervalMs);
        connect(timer_, &QTimer::timeout, this, &FrankaPoseClient::poll);
        connect(&network_, &QNetworkAccessManager::finished, this, &FrankaPoseClient::handleReply);
    }

    static constexpr int kPollIntervalMs   = 100;   // 10 Hz; the arm creeps at 0.03 m/s
    static constexpr double kStaleAfterSeconds = 0.5;

public slots:
    // `url` is the server root, e.g. "http://100.93.60.35:8732".
    void start(const QString &url) {
        url_ = url;
        while (url_.endsWith(QLatin1Char('/'))) {
            url_.chop(1);
        }
        active_ = true;
        emit statusChanged(false, QStringLiteral("connecting to %1 ...").arg(url_));
        poll();
        timer_->start();
    }

    void stop() {
        active_ = false;
        timer_->stop();
        emit statusChanged(false, QStringLiteral("off"));
    }

    bool active() const {
        return active_;
    }

signals:
    // Robot base XY in millimetres, exactly as the Franka reports it.
    void poseReady(double xMm, double yMm, double zMm);
    // Emitted whenever the link's state changes meaning, so the pane never shows a
    // number with no indication of whether it is live.
    void statusChanged(bool ok, QString message);

private slots:
    void poll() {
        if (!active_ || in_flight_) {
            return;   // one request at a time: a stalled tailnet must not queue up
        }
        in_flight_ = true;
        QNetworkRequest request{QUrl(url_ + QStringLiteral("/pose"))};
        QNetworkReply *reply = network_.get(request);
        // QNetworkRequest::setTransferTimeout is Qt 5.15; this project builds
        // against 5.12, so the watchdog is explicit. Without one, a tailnet that
        // goes away mid-request leaves in_flight_ set forever and the readout
        // freezes on its last value while still looking live.
        QTimer::singleShot(kPollIntervalMs * 4, reply, [reply]() {
            if (reply->isRunning()) {
                reply->abort();
            }
        });
    }

    void handleReply(QNetworkReply *reply) {
        in_flight_ = false;
        const QByteArray body = reply->readAll();
        const QNetworkReply::NetworkError error = reply->error();
        reply->deleteLater();
        if (!active_) {
            return;
        }

        if (error != QNetworkReply::NoError && body.isEmpty()) {
            reportProblem(QStringLiteral("unreachable (%1) -- is pose_server.py running on tactile?")
                              .arg(reply->errorString()));
            return;
        }
        const QJsonObject payload = QJsonDocument::fromJson(body).object();
        if (!payload.value(QStringLiteral("ok")).toBool()) {
            reportProblem(payload.value(QStringLiteral("error")).toString(QStringLiteral("bad reply")));
            return;
        }
        const double age = payload.value(QStringLiteral("age_s")).toDouble(0.0);
        if (age > kStaleAfterSeconds) {
            reportProblem(QStringLiteral("pose is %1 s stale -- franka_states has stopped "
                                         "(user-stop pressed? reflex latched?)")
                              .arg(age, 0, 'f', 1));
            return;
        }

        if (!last_ok_) {
            last_ok_ = true;
            emit statusChanged(true, QStringLiteral("live from %1").arg(url_));
        }
        emit poseReady(payload.value(QStringLiteral("x_mm")).toDouble(),
                       payload.value(QStringLiteral("y_mm")).toDouble(),
                       payload.value(QStringLiteral("z_mm")).toDouble());
    }

private:
    void reportProblem(const QString &message) {
        if (last_ok_ || last_message_ != message) {
            last_ok_       = false;
            last_message_  = message;
            emit statusChanged(false, message);
        }
    }

    QNetworkAccessManager network_;
    QTimer *timer_    = nullptr;
    QString url_;
    QString last_message_;
    bool active_    = false;
    bool in_flight_ = false;
    bool last_ok_   = false;
};

} // namespace e_bts::gui

#endif // E_BTS_GUI_FRANKA_POSE_CLIENT_H
