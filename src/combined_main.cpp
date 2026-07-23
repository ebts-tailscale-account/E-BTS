// E_BTS_GUI entry point. Bootstraps the Qt Quick application: a single
// CameraSessionWorker on its own thread owns the shared Metavision::Camera
// and everything that consumes it, while Main.qml renders the ribbon and
// whichever of the Camera / Circle Tracking / Sequence Recording panes are
// open (see gui/camera_session_worker.h for why there is exactly one camera
// session shared by all three).

#include <QGuiApplication>
#include <QIcon>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QThread>
#include <QUrl>

#include "gui/camera_session_worker.h"
#include "gui/frame_view.h"
#include "gui/gui_bridge.h"

int main(int argc, char *argv[]) {
    QQuickStyle::setStyle("Material");

    QGuiApplication app(argc, argv);
    app.setApplicationName("E-BTS GUI");
    app.setOrganizationName("Tactile Lab, Nazarbayev University");
    app.setWindowIcon(QIcon(QStringLiteral(":/icon.png")));

    qmlRegisterType<e_bts::gui::FrameView>("EBts", 1, 0, "FrameView");

    auto *worker      = new e_bts::gui::CameraSessionWorker();
    auto *workerThread = new QThread();
    worker->moveToThread(workerThread);
    QObject::connect(workerThread, &QThread::finished, worker, &QObject::deleteLater);
    workerThread->start();

    // Lives on the main thread (unlike worker), since QML's Connections type
    // refuses to bind to a target object living in a different thread than
    // the QQmlEngine -- see gui/gui_bridge.h.
    auto *bridge = new e_bts::gui::GuiBridge(worker);

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("cameraWorker", worker);
    engine.rootContext()->setContextProperty("cameraEvents", bridge);
    engine.load(QUrl(QStringLiteral("qrc:/qml/Main.qml")));
    if (engine.rootObjects().isEmpty()) {
        return -1;
    }

    // Deferred until the worker thread's event loop is actually running so
    // the very first slot call goes through the same queued-connection path
    // every later one does.
    QMetaObject::invokeMethod(worker, "connectToCamera", Qt::QueuedConnection);

    const int exit_code = app.exec();

    workerThread->quit();
    workerThread->wait();
    delete workerThread;

    return exit_code;
}
