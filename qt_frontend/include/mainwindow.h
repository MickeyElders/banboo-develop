#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QStackedWidget>
#include <QtQuickWidgets/QQuickWidget>
#include <QTimer>
#include <memory>

class VideoRenderer;
class TouchController;
class CameraManager;
class BambooDetector;
class SystemController;
class ConfigManager;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void resizeEvent(QResizeEvent *event) override;
    void closeEvent(QCloseEvent *event) override;

private slots:
    void onStartButtonClicked();
    void onStopButtonClicked();
    void onSettingsButtonClicked();
    void onSystemStatusUpdate();
    void onDetectionResult(const QRect& bambooRect, float confidence);
    void onCameraFrameReady();

private:
    void setupUI();
    void setupConnections();
    void updateStatusBar();
    void switchToOperationMode();
    void switchToStandbyMode();

    // UI组件
    QWidget *m_centralWidget;
    QVBoxLayout *m_mainLayout;
    QHBoxLayout *m_topLayout;
    QHBoxLayout *m_bottomLayout;
    
    QQuickWidget *m_videoWidget;
    QPushButton *m_startButton;
    QPushButton *m_stopButton;
    QPushButton *m_settingsButton;
    
    QLabel *m_statusLabel;
    QLabel *m_frameRateLabel;
    QLabel *m_detectionLabel;
    QProgressBar *m_progressBar;
    
    QStackedWidget *m_stackedWidget;

    // 核心组件
    std::unique_ptr<VideoRenderer> m_videoRenderer;
    std::unique_ptr<TouchController> m_touchController;
    std::unique_ptr<CameraManager> m_cameraManager;
    std::unique_ptr<BambooDetector> m_bambooDetector;
    std::unique_ptr<SystemController> m_systemController;
    std::unique_ptr<ConfigManager> m_configManager;

    // 定时器
    QTimer *m_statusTimer;
    
    // 状态
    bool m_isRunning;
    int m_frameCount;
    qint64 m_lastFrameTime;
};

#endif // MAINWINDOW_H