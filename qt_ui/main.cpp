#include <QCoreApplication>
#include <QDir>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QProcessEnvironment>
#include <QUrl>

#include "DeepStreamRunner.h"
#include "ModbusClient.h"
#include "StateModels.h"

namespace {
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
}  // namespace

int main(int argc, char *argv[]) {
    configureHeadlessEnvironment();

    QGuiApplication app(argc, argv);
    QCoreApplication::setApplicationName("AI竹节识别切割系统");

    SystemState systemState;
    JetsonState jetsonState;
    AiState aiState;
    WifiState wifiState;
    DeepStreamRunner deepStream;
    ModbusClient modbus;

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
