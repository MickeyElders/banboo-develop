#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include "StateModels.h"
#include "DeepStreamStub.h"
#include "ModbusClient.h"

int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);
    QCoreApplication::setApplicationName("AI竹节识别切割系统");

    SystemState systemState;
    JetsonState jetsonState;
    AiState aiState;
    WifiState wifiState;
    DeepStreamStub deepStream;
    ModbusClient modbus;

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("systemState", &systemState);
    engine.rootContext()->setContextProperty("jetsonState", &jetsonState);
    engine.rootContext()->setContextProperty("aiState", &aiState);
    engine.rootContext()->setContextProperty("wifiState", &wifiState);
    engine.rootContext()->setContextProperty("deepStream", &deepStream);
    engine.rootContext()->setContextProperty("modbus", &modbus);

    const QUrl url(u"qrc:/Main.qml"_qs);
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated, &app,
                     [url](QObject *obj, const QUrl &objUrl) {
                         if (!obj && url == objUrl)
                             QCoreApplication::exit(-1);
                     }, Qt::QueuedConnection);
    engine.load(url);
    return app.exec();
}
