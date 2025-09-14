#include <QtWidgets/QApplication>
#include <QtCore/QDir>
#include <QtCore/QStandardPaths>
#include <QtCore/QLoggingCategory>
#include <QtCore/QCommandLineParser>
#include <QtCore/QTranslator>
#include <QtCore/QLibraryInfo>
#include <QtQml/qqml.h>
#include <QQuickStyle>
#include <QFont>
#include <QStyleFactory>

#include "mainwindow.h"
#include "configmanager.h"
#include "bamboodetector.h"

// 日志分类
Q_LOGGING_CATEGORY(appMain, "app.main")
Q_LOGGING_CATEGORY(appUI, "app.ui")
Q_LOGGING_CATEGORY(appCamera, "app.camera")
Q_LOGGING_CATEGORY(appDetection, "app.detection")

void setupLogging()
{
    // 设置日志格式
    qSetMessagePattern("[%{time yyyy-MM-dd hh:mm:ss.zzz}] "
                      "[%{category}] "
                      "[%{if-debug}D%{endif}%{if-info}I%{endif}%{if-warning}W%{endif}%{if-critical}C%{endif}%{if-fatal}F%{endif}] "
                      "%{message} "
                      "(%{file}:%{line})");

    // 创建日志目录
    QString logDir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/logs";
    QDir().mkpath(logDir);
}

void setupApplication(QApplication &app)
{
    // 应用程序信息
    app.setApplicationName("Bamboo Controller Qt");
    app.setApplicationVersion("2.0.0");
    app.setOrganizationName("BambooTech");
    app.setOrganizationDomain("bambootech.com");
    app.setApplicationDisplayName("智能切竹机控制系统");

    // 设置应用程序图标
    app.setWindowIcon(QIcon(":/icons/app_icon.png"));

    // 为触摸设备优化
    app.setAttribute(Qt::AA_SynthesizeTouchForUnhandledMouseEvents, true);
    app.setAttribute(Qt::AA_SynthesizeMouseForUnhandledTouchEvents, true);
    
    // 启用高DPI支持
    app.setAttribute(Qt::AA_EnableHighDpiScaling, true);
    app.setAttribute(Qt::AA_UseHighDpiPixmaps, true);

    // 设置字体
    QFont font = app.font();
    font.setFamily("Noto Sans CJK SC");
    font.setPointSize(12);
    app.setFont(font);

    // 设置样式
    QQuickStyle::setStyle("Material");
    
    // 注册元对象类型
    qmlRegisterType<BambooDetector>("BambooDetector", 1, 0, "BambooDetector");
}

void setupTranslations(QApplication &app)
{
    QTranslator *qtTranslator = new QTranslator(&app);
    if (qtTranslator->load("qt_zh_CN", QLibraryInfo::location(QLibraryInfo::TranslationsPath))) {
        app.installTranslator(qtTranslator);
    }

    QTranslator *appTranslator = new QTranslator(&app);
    if (appTranslator->load(":/translations/bamboo_controller_zh_CN.qm")) {
        app.installTranslator(appTranslator);
    }
}

bool checkSystemRequirements()
{
    // 检查OpenGL ES支持
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGLES);
    format.setVersion(2, 0);
    QSurfaceFormat::setDefaultFormat(format);

    // 检查必要的库文件
    QStringList requiredLibs = {
        "libopencv_core.so",
        "libopencv_imgproc.so",
        "libopencv_highgui.so",
        "libopencv_dnn.so"
    };

    for (const QString &lib : requiredLibs) {
        if (!QLibrary::isLibrary(lib)) {
            qCCritical(appMain) << "Required library not found:" << lib;
            return false;
        }
    }

    return true;
}

int main(int argc, char *argv[])
{
    // 启用OpenGL ES
    qputenv("QT_QUICK_BACKEND", "opengl");
    qputenv("QSG_RHI_BACKEND", "opengl");
    
    QApplication app(argc, argv);

    // 设置日志
    setupLogging();
    qCInfo(appMain) << "Bamboo Controller Qt starting...";

    // 解析命令行参数
    QCommandLineParser parser;
    parser.setApplicationDescription("智能切竹机控制系统 - Qt前端");
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption configOption(QStringList() << "c" << "config",
                                   "配置文件路径 <file>.",
                                   "file");
    parser.addOption(configOption);

    QCommandLineOption debugOption(QStringList() << "d" << "debug",
                                  "启用调试模式");
    parser.addOption(debugOption);

    QCommandLineOption fullscreenOption(QStringList() << "f" << "fullscreen",
                                       "全屏模式");
    parser.addOption(fullscreenOption);

    parser.process(app);

    // 检查系统要求
    if (!checkSystemRequirements()) {
        qCCritical(appMain) << "System requirements not met!";
        return -1;
    }

    // 设置应用程序
    setupApplication(app);
    setupTranslations(app);

    // 初始化配置管理器
    ConfigManager *configManager = new ConfigManager(&app);
    QString configPath = parser.value(configOption);
    if (!configManager->initialize(configPath)) {
        qCCritical(appMain) << "Failed to initialize configuration:" << configManager->lastError();
        return -1;
    }

    // 应用配置
    if (parser.isSet(debugOption)) {
        QLoggingCategory::setFilterRules("*.debug=true");
    }

    // 创建主窗口
    MainWindow window;
    
    // 应用UI配置
    ConfigManager::UIConfig uiConfig = configManager->uiConfig();
    if (parser.isSet(fullscreenOption) || uiConfig.fullscreen) {
        window.showFullScreen();
    } else {
        window.resize(uiConfig.windowWidth, uiConfig.windowHeight);
        window.show();
    }

    qCInfo(appMain) << "Application started successfully";

    // 运行应用程序
    int result = app.exec();

    qCInfo(appMain) << "Application exiting with code:" << result;
    return result;
}