#include <QCoreApplication>
#include <QDir>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QProcessEnvironment>
#include <QFile>
#include <QFileDevice>
#include <QLoggingCategory>
#include <QTimer>
#include <QUrl>
#include <atomic>
#include <csignal>

#include "DeepStreamRunner.h"
#include "ModbusClient.h"
#include "StateModels.h"
#include "WebPreview.h"

namespace {
std::atomic_bool g_shouldQuit{false};

void handleSignal(int signum) {
    if (signum == SIGTERM || signum == SIGINT) {
        g_shouldQuit.store(true);
    }
}

void configureHeadlessEnvironment() {
    // Always offscreen for headless stability; RTSP承担可视输出
    qputenv("QT_QPA_PLATFORM", "offscreen");
    qputenv("QT_QPA_EGLFS_INTEGRATION", "");
    qputenv("QT_QPA_EGLFS_KMS_CONFIG", "");
    qputenv("EGL_PLATFORM", "surfaceless");
    qputenv("EGL_DISPLAY", "surfaceless");
    if (qEnvironmentVariableIsEmpty("XDG_RUNTIME_DIR")) {
        const QByteArray runtimeDir("/run/bamboo-qt");
        qputenv("XDG_RUNTIME_DIR", runtimeDir);
        QDir().mkpath(QString::fromUtf8(runtimeDir));
    }
    // Enforce 0700 permissions on runtime dir to satisfy QStandardPaths
    const QString rd = QString::fromUtf8(qgetenv("XDG_RUNTIME_DIR"));
    if (!rd.isEmpty()) {
        QFile::setPermissions(rd, QFile::Permissions(
            QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner));
    }
}

void logOffscreenEnvironment() {
    qInfo() << "[startup] QT_QPA_PLATFORM =" << qgetenv("QT_QPA_PLATFORM");
    qInfo() << "[startup] XDG_RUNTIME_DIR =" << qgetenv("XDG_RUNTIME_DIR");
    qInfo() << "[startup] EGL_PLATFORM =" << qgetenv("EGL_PLATFORM");
}
}  // namespace

int main(int argc, char *argv[]) {
    // Enable verbose Qt logs for diagnostics
    QLoggingCategory::setFilterRules(QStringLiteral(
        "qt.qpa.*=true\n"
        "qt.scenegraph.general=true\n"
        "qt.rhi.*=true\n"
        "qt.quick.*=true\n"));
    std::signal(SIGTERM, handleSignal);
    std::signal(SIGINT, handleSignal);
    configureHeadlessEnvironment();
    logOffscreenEnvironment();

    QGuiApplication app(argc, argv);
    app.setQuitOnLastWindowClosed(false);
    QCoreApplication::setApplicationName("AI Bamboo");
    qInfo() << "[startup] QGuiApplication constructed";

    qInfo() << "[startup] Constructing core state objects...";
    SystemState systemState;
    JetsonState jetsonState;
    AiState aiState;
    WifiState wifiState;
    qInfo() << "[startup] Core state objects OK, constructing pipelines...";
    DeepStreamRunner deepStream;
    qInfo() << "[startup] DeepStreamRunner constructed";
    ModbusClient modbus;
    qInfo() << "[startup] ModbusClient constructed";

    QTimer quitTimer;
    quitTimer.setInterval(200);
    QObject::connect(&quitTimer, &QTimer::timeout, &app, [&app]() {
        if (g_shouldQuit.load()) {
            app.quit();
        }
    });
    quitTimer.start();

    QObject::connect(&app, &QCoreApplication::aboutToQuit, &deepStream, [&deepStream]() { deepStream.stop(); });
    QObject::connect(&app, &QCoreApplication::aboutToQuit, &modbus, [&modbus]() { modbus.shutdown(); });

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("systemState", &systemState);
    engine.rootContext()->setContextProperty("jetsonState", &jetsonState);
    engine.rootContext()->setContextProperty("aiState", &aiState);
    engine.rootContext()->setContextProperty("wifiState", &wifiState);
    engine.rootContext()->setContextProperty("deepStream", &deepStream);
    engine.rootContext()->setContextProperty("modbus", &modbus);

    qInfo() << "[startup] Loading QML...";
    const QUrl url(QStringLiteral("qrc:/Main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated, &app,
                     [url](QObject *obj, const QUrl &objUrl) {
                         if (!obj && url == objUrl) {
                             QCoreApplication::exit(-1);
                         }
                     },
                     Qt::QueuedConnection);
    engine.load(url);
    if (engine.rootObjects().isEmpty()) {
        qCritical() << "[startup] Failed to load QML root" << url;
        return -1;
    }

    QQuickWindow *rootWindow = qobject_cast<QQuickWindow *>(engine.rootObjects().first());
    // Serve bamboo.html (installed to prefix) with embedded MJPEG at /mjpeg
    const QString htmlPath = QCoreApplication::applicationDirPath() + "/../bamboo.html";
    WebPreview preview(rootWindow, htmlPath, 8080, &app);
    qInfo() << "[startup] Web preview on http://<device-ip>:8080/ (MJPEG at /mjpeg)";

    qInfo() << "[startup] QML loaded, entering event loop";
    return app.exec();
}
