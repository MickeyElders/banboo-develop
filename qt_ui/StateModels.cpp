#include "StateModels.h"
#include <QtGlobal>
#include <QRandomGenerator>

SystemState::SystemState(QObject *parent) : QObject(parent) {
    connect(&m_heartbeatTimer, &QTimer::timeout, this, [this]() {
        m_heartbeat = (m_heartbeat + 1) % 4000000000;
        Q_EMIT stateChanged();
    });
    connect(&m_workflowTimer, &QTimer::timeout, this, &SystemState::simulateStep);
    m_heartbeatTimer.start(500); // 500ms heartbeats
}

void SystemState::start() {
    if (m_running) return;
    m_running = true;
    m_currentStep = 1;
    m_currentProcess = "Feeding";
    m_workflowTimer.start(800);
    Q_EMIT stateChanged();
}

void SystemState::pause() {
    m_running = false;
    m_workflowTimer.stop();
    Q_EMIT stateChanged();
}

void SystemState::stop() {
    m_running = false;
    m_workflowTimer.stop();
    m_currentStep = 1;
    m_currentProcess = "Stopped";
    Q_EMIT stateChanged();
}

void SystemState::emergencyStop() {
    m_running = false;
    m_workflowTimer.stop();
    m_currentProcess = "Emergency Stop";
    Q_EMIT stateChanged();
}

void SystemState::simulateStep() {
    if (!m_running) return;
    m_currentStep = (m_currentStep % 5) + 1;
    switch (m_currentStep) {
        case 1: m_currentProcess = "Feeding"; break;
        case 2: m_currentProcess = "Vision"; break;
        case 3: m_currentProcess = "Coordinate TX"; break;
        case 4: m_currentProcess = "Cut Prep"; break;
        case 5: m_currentProcess = "Cutting"; break;
    }
    m_xCoordinate = 200 + QRandomGenerator::global()->bounded(600.0);
    m_cutQuality = (QRandomGenerator::global()->bounded(100) > 5) ? "OK" : "FAIL";
    Q_EMIT stateChanged();
}

JetsonState::JetsonState(QObject *parent) : QObject(parent) {
    connect(&m_timer, &QTimer::timeout, this, [this]() {
        m_cpuUsage = 35 + QRandomGenerator::global()->bounded(30.0);
        m_gpuUsage = 20 + QRandomGenerator::global()->bounded(35.0);
        m_memUsed = 1.8 + QRandomGenerator::global()->bounded(3.0);
        m_temp = 45 + QRandomGenerator::global()->bounded(12.0);
        m_fanRpm = 1800 + QRandomGenerator::global()->bounded(600.0);
        Q_EMIT changed();
    });
    m_timer.start(2000);
}

void JetsonState::setPerfMode(const QString &mode) {
    m_perfMode = mode;
    Q_EMIT changed();
}

AiState::AiState(QObject *parent) : QObject(parent) {
    connect(&m_timer, &QTimer::timeout, this, [this]() {
        m_inferenceMs = 12 + QRandomGenerator::global()->bounded(8.0);
        m_fps = 24 + QRandomGenerator::global()->bounded(8.0);
        m_today += 1;
        m_total += 1;
        Q_EMIT changed();
    });
    m_timer.start(3000);
}

WifiState::WifiState(QObject *parent) : QObject(parent) {}

void WifiState::apply(const QString &ssid, const QString &password, const QString &mode,
                      const QString &ip, const QString &mask, const QString &gw, const QString &dns) {
    Q_UNUSED(password)
    Q_UNUSED(ip)
    Q_UNUSED(mask)
    Q_UNUSED(gw)
    Q_UNUSED(dns)
    m_ssid = ssid;
    m_mode = mode.toUpper();
    m_status = (mode.toLower() == "dhcp") ? "DHCP acquiring" : "Static applied";
    Q_EMIT changed();
}

void WifiState::check() {
    const bool ok = QRandomGenerator::global()->bounded(100) > 15;
    m_status = ok ? "Connected" : "Disconnected";
    m_rssi = ok ? -50 - QRandomGenerator::global()->bounded(15) : -120;
    Q_EMIT changed();
}
