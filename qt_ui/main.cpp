#include <QCoreApplication>
#include <QDir>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QProcessEnvironment>
#include <QTimer>
#include <QUrl>
#include <QFile>
#include <QDirIterator>
#include <QTextStream>
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
    // Prefer eglfs/DRM when DISPLAY is absent; allow user overrides when set.
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM") && qEnvironmentVariableIsEmpty("DISPLAY")) {
        qputenv("QT_QPA_PLATFORM", "eglfs");
    }
    if (qEnvironmentVariableIsEmpty("QT_QPA_EGLFS_INTEGRATION")) {
        qputenv("QT_QPA_EGLFS_INTEGRATION", "eglfs_kms");
    }
    if (qEnvironmentVariableIsEmpty("EGL_PLATFORM")) {
        qputenv("EGL_PLATFORM", "drm");
    }
    if (qEnvironmentVariableIsEmpty("XDG_RUNTIME_DIR")) {
        const QByteArray runtimeDir("/run/bamboo-qt");
        qputenv("XDG_RUNTIME_DIR", runtimeDir);
        QDir().mkpath(QString::fromUtf8(runtimeDir));
    }
}

void logKmsEnvironment() {
    qInfo() << "[startup] QT_QPA_PLATFORM =" << qgetenv("QT_QPA_PLATFORM");
    qInfo() << "[startup] QT_QPA_EGLFS_INTEGRATION =" << qgetenv("QT_QPA_EGLFS_INTEGRATION");
    qInfo() << "[startup] QT_QPA_EGLFS_KMS_CONFIG =" << qgetenv("QT_QPA_EGLFS_KMS_CONFIG");
    QFile cfg(QString::fromUtf8(qgetenv("QT_QPA_EGLFS_KMS_CONFIG")));
    if (cfg.exists()) {
        qInfo() << "[startup] Found eglfs_kms config:" << cfg.fileName();
    } else {
        qWarning() << "[startup] Missing eglfs_kms config file:" << cfg.fileName();
    }
    QStringList cards;
    QDirIterator it("/dev/dri", QDir::System | QDir::Readable | QDir::NoDotAndDotDot);
    while (it.hasNext()) {
        const QString path = it.next();
        if (path.contains("card"))
            cards << path;
    }
    qInfo() << "[startup] DRM devices:" << cards.join(",");
}
}  // namespace

int main(int argc, char *argv[]) {
    std::signal(SIGTERM, handleSignal);
    std::signal(SIGINT, handleSignal);
    configureHeadlessEnvironment();
    logKmsEnvironment();

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
    return app.exec();
}
