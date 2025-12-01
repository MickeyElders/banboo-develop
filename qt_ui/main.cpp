#include <QCoreApplication>
#include <QDir>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QProcessEnvironment>
#include <QFile>
#include <QTimer>
#include <QUrl>
#include <atomic>
#include <csignal>

#include "DeepStreamRunner.h"
#include "ModbusClient.h"
#include "StateModels.h"

namespace {
std::atomic_bool g_shouldQuit{false};

void handleSignal(int signum) {
    if (signum == SIGTERM || signum == SIGINT) {
        g_shouldQuit.store(true);
    }
}

void configureHeadlessEnvironment() {
    // Force offscreen; KMS path is disabled due to JetPack 6.x headless connector issues.
    qputenv("QT_QPA_PLATFORM", "offscreen");
    qputenv("QT_QPA_EGLFS_INTEGRATION", "");
    qputenv("QT_QPA_EGLFS_KMS_CONFIG", "");
    qputenv("EGL_PLATFORM", "");
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
}
}  // namespace

int main(int argc, char *argv[]) {
    std::signal(SIGTERM, handleSignal);
    std::signal(SIGINT, handleSignal);
    configureHeadlessEnvironment();
    logOffscreenEnvironment();

    QGuiApplication app(argc, argv);
    app.setQuitOnLastWindowClosed(false);
    QCoreApplication::setApplicationName("AI Bamboo");

    SystemState systemState;
    JetsonState jetsonState;
    AiState aiState;
    WifiState wifiState;
    DeepStreamRunner deepStream;
    ModbusClient modbus;

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
    qInfo() << "[startup] QML loaded, entering event loop";
    return app.exec();
}
