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
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <gst/gst.h>
#include <atomic>
#include <csignal>

#include "DeepStreamRunner.h"
#include "ModbusClient.h"
#include "StateModels.h"
#include "WebPreview.h"
#include "WebRTCSignaling.h"
#include <iostream>

namespace {
std::atomic_bool g_shouldQuit{false};

// Pre-initialize a surfaceless EGL context to unblock nvargus/gst when headless.
void fixArgusDeadlockInHeadless() {
    // Pre-start nvargus-daemon to avoid lazy startup stalls
    (void)system("systemctl start nvargus-daemon.service >/dev/null 2>&1");

    // Try surfaceless first; fall back to default display.
    EGLDisplay display = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, nullptr, nullptr);
    if (display == EGL_NO_DISPLAY) {
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    }
    EGLint major = 0, minor = 0;
    if (eglInitialize(display, &major, &minor) == EGL_FALSE) {
        return;  // likely already initialized elsewhere
    }
    eglBindAPI(EGL_OPENGL_ES_API);
    static const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };
    EGLConfig config;
    EGLint numConfigs = 0;
    eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);
    static const EGLint contextAttribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
    if (context != EGL_NO_CONTEXT) {
        EGLint pbufferAttribs[] = {
            EGL_WIDTH, 16,
            EGL_HEIGHT, 16,
            EGL_NONE,
        };
        EGLSurface surface = eglCreatePbufferSurface(display, config, pbufferAttribs);
        eglMakeCurrent(display, surface, surface, context);
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroySurface(display, surface);
        eglDestroyContext(display, context);
    }
    eglTerminate(display);

    // Force GStreamer GL to use EGL/GLES2
    g_setenv("GST_GL_PLATFORM", "egl", TRUE);
    g_setenv("GST_GL_API", "gles2", TRUE);
}

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
    fixArgusDeadlockInHeadless();  // must run before any gst_init in DeepStream
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
    QObject::connect(&deepStream, &DeepStreamRunner::errorChanged, [](const QString &msg) {
        qCritical() << "[deepstream] error:" << msg;
    });

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
    // 默认关闭 MJPEG 预览，设置 WEB_PREVIEW=1 可开启（避免与 WebRTC/其他端口冲突）
    bool enablePreview = false;
    bool okEnv = false;
    int previewEnv = qEnvironmentVariableIntValue("WEB_PREVIEW", &okEnv);
    if (okEnv && previewEnv == 1) {
        enablePreview = true;
    }
    if (!enablePreview) {
        qInfo() << "[startup] Web preview disabled (set WEB_PREVIEW=1 to enable)";
    }
    if (enablePreview) {
        const QString htmlPath = QCoreApplication::applicationDirPath() + "/../bamboo.html";
        WebPreview preview(rootWindow, htmlPath, 8080, &app);
        qInfo() << "[startup] Web preview on http://<device-ip>:8080/ (MJPEG at /mjpeg)";
    }

    // Start WebRTC signaling server; DeepStream webrtcbin will use it.
    WebRTCSignaling signaling(8080, &app);
    deepStream.setWebRTCSignaling(&signaling);

    // Trigger DeepStream autostart here (instead of constructor) for clearer logs
    QTimer::singleShot(0, [&deepStream]() {
        const QByteArray autoStart = qgetenv("DS_AUTOSTART");
        if (autoStart == "1") {
            std::cout << "[deepstream] Autostart from main()" << std::endl;
            const bool ok = deepStream.start({});
            std::cout << "[deepstream] Autostart result: " << (ok ? "success" : "failure") << std::endl;
        } else {
            std::cout << "[deepstream] Autostart disabled (DS_AUTOSTART!=1)" << std::endl;
        }
    });

    qInfo() << "[startup] QML loaded, entering event loop";
    return app.exec();
}
