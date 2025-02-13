

#include "gpu_performance_monitor.h"
#include <memory>
#include <qboxlayout.h>
#include <qchartview.h>
#include <qcursor.h>
#include <qdatetime.h>
#include <qdatetimeedit.h>
#include <qfuturewatcher.h>
#include <qjsonobject.h>
#include <qlabel.h>
#include <qline.h>
#include <qlineseries.h>
#include <qmutex.h>
#include <qnamespace.h>
#include <qobject.h>
#include <qobjectdefs.h>
#include <qpainter.h>
#include <qpoint.h>
#include <qtabwidget.h>
#include <qtooltip.h>
#include <qvalueaxis.h>
#include <qwidget.h>

#include "../../dbconn.h"


gpu_performance_monitor::gpu_performance_monitor(QWidget* parent)
    : QWidget(parent)
    { 
    QVBoxLayout* monitor_wrapper = new QVBoxLayout(this);

    m_train_log_websocket = new get_train_log_websocket(this);

    m_train_log_websocket->connect_to_server(QUrl("ws://localhost:8765"));

    QTabWidget* monitor_tabs = new QTabWidget();
    QWidget* gpu_monitor_tab = new QWidget();

    QVBoxLayout* gpu_monitor_wrapper = new QVBoxLayout();
    QVBoxLayout* train_monitor_wrapper = new QVBoxLayout();

    QHBoxLayout* monitor_thermal_status_wrapper = new QHBoxLayout();

    QHBoxLayout* monitor_usage_status_wrapper = new QHBoxLayout();

    m_temp_status = new QLabel("GPU Temp: N/A");
    m_power_status = new QLabel("GPU Power: N/A");
    
    m_mem_status = new QLabel("Memory Usage: N/A");
    m_gpu_util_status = new QLabel("GPU Usage: N/A");

    monitor_thermal_status_wrapper->addWidget(m_temp_status);
    monitor_thermal_status_wrapper->addWidget(m_power_status);

    monitor_usage_status_wrapper->addWidget(m_mem_status);
    monitor_usage_status_wrapper->addWidget(m_gpu_util_status);
    
    gpu_monitor_wrapper->addLayout(monitor_thermal_status_wrapper);
    gpu_monitor_wrapper->addLayout(monitor_usage_status_wrapper);

    m_thermal_x_axis = new QValueAxis();
    m_thermal_x_axis->setTitleText("Time (s)");
    m_thermal_x_axis->setRange(0, 60);

    m_y_temp = new QValueAxis();
    m_y_temp->setRange(0, 100);
    m_y_temp->setTitleText("Temperature (°C)");
    
    m_y_power = new QValueAxis();
    m_y_power->setRange(0, 370);
    m_y_power->setTitleText("Power (W)");

    m_thermal_chart = new QChart();
    m_thermal_chart->setTitle("GPU Monitor");
    m_thermal_chart->legend()->setAlignment(Qt::AlignBottom);
    m_thermal_chart->setAnimationOptions(QChart::SeriesAnimations);
    
    m_thermal_chart->addAxis(m_thermal_x_axis, Qt::AlignBottom);
    m_thermal_chart->addAxis(m_y_temp, Qt::AlignLeft);
    m_thermal_chart->addAxis(m_y_power, Qt::AlignRight);
    
    m_thermal_view = new QChartView(m_thermal_chart);
    m_thermal_view->setRenderHint(QPainter::Antialiasing);

    m_usage_x_axis = new QValueAxis();
    m_usage_x_axis->setTitleText("Time (s)");
    m_usage_x_axis->setRange(0, 60);
    
    m_y_mem = new QValueAxis();
    m_y_mem->setRange(0, 100);
    m_y_mem->setTitleText("Usage (%)");

    m_y_gpu_util = new QValueAxis();
    m_y_gpu_util->setRange(0, 100);
    m_y_gpu_util->setTitleText("GPU Usage (%)");

    m_usage_chart = new QChart();
    m_usage_chart->setTitle("GPU Monitor");
    m_usage_chart->legend()->setAlignment(Qt::AlignBottom);
    m_usage_chart->setAnimationOptions(QChart::SeriesAnimations);

    m_usage_chart->addAxis(m_usage_x_axis, Qt::AlignBottom);
    m_usage_chart->addAxis(m_y_mem, Qt::AlignLeft);
    m_usage_chart->addAxis(m_y_gpu_util, Qt::AlignRight);
    
    m_usage_view = new QChartView(m_usage_chart);
    m_usage_view->setRenderHint(QPainter::Antialiasing);

    gpu_monitor_wrapper->addWidget(m_usage_view);
    gpu_monitor_wrapper->addWidget(m_thermal_view);
    gpu_monitor_tab->setLayout(gpu_monitor_wrapper);

    QWidget* training_log_tab = new QWidget();

    m_training_log_chart = new QChart();
    m_train_x_axis = new QValueAxis();
    m_train_y_loss = new QValueAxis();
    m_train_y_lr = new QValueAxis();

    m_train_x_axis->setTitleText("Epoch");
    m_train_y_loss->setTitleText("Loss");
    m_train_y_lr->setTitleText("Lr");

    m_training_log_chart->addAxis(m_train_x_axis, Qt::AlignBottom);
    m_training_log_chart->addAxis(m_train_y_loss, Qt::AlignLeft);
    m_training_log_chart->addAxis(m_train_y_lr, Qt::AlignRight);



    m_training_log_view = new QChartView(m_training_log_chart);
    m_training_log_view->setRenderHint(QPainter::Antialiasing);
    train_monitor_wrapper->addWidget(m_training_log_view);
    training_log_tab->setLayout(train_monitor_wrapper);

    monitor_tabs->addTab(gpu_monitor_tab, "gpu");
    monitor_tabs->addTab(training_log_tab, "training");

    
    monitor_wrapper->addWidget(monitor_tabs);

    init_monitors();
    init_training_monitors();
    m_nvml_initialized = init_NVML();

    if (!m_nvml_initialized) {
        QLabel* nvml_init_failed_label = new QLabel("⚠️ GPU Monitoring Unvailable!");
        nvml_init_failed_label->setStyleSheet("color: red; font-weight: bold;");
        gpu_monitor_wrapper->addWidget(nvml_init_failed_label);
        return;
    }
    m_update_timer = new QTimer(this);
    m_start_time = QDateTime::currentSecsSinceEpoch();

    connect(m_update_timer, &QTimer::timeout, this, [this]() {
        if (!m_data_watcher.isRunning()) {
            m_data_watcher.setFuture(QtConcurrent::run([this](){ data_collection_thread(); }));
        }
    });

    connect ( m_train_log_websocket
            , &get_train_log_websocket::train_started
            , this
            , [this](QString train_id, int total_batches, int total_epochs) {
                QString current_train_id = train_id;
                int current_total_batches = total_batches;
                int current_total_epochs = total_epochs;

                qDebug() << "train_id: " << current_train_id;
                qDebug() << "total_batches: " << current_total_batches;
                qDebug() << "total_epochs: " << current_total_epochs;
            });

    connect ( m_train_log_websocket
            , &get_train_log_websocket::train_ended
            , this
            , [this](QJsonObject report) {
                const QString train_id = report.value("train_id").toString();
                const QJsonObject model_info = report.value("model_info").toObject();
                qDebug() << "Training is over!/n/n/n";
                qDebug().nospace()
                    << "\n=== Training Report ==="
                    << "\nTrain ID: " << train_id
                    << "\nParameters: " << model_info.value("params").toVariant().toLongLong()
                    << "\nLayers: " << model_info.value("layers").toInt()
                    << "\nmAP50: " << report["val_metrics"].toObject().value("mAP50").toDouble()
                    << "\nDuration: " << report.value("duration").toString();
                
                dbConn::instance().insert_new_trained_model(report);
            });

    connect ( m_train_log_websocket
            , &get_train_log_websocket::training_metrics
            , this
            , [this](QJsonObject metrics) {
            int epoch = metrics["epoch"].toInt();
            float loss = metrics["loss"].toDouble();
            float lr = metrics["lr"].toDouble();

            static float current_loss = 0.f;
            current_loss = loss;
            qDebug() << "loss: " << current_loss;
            m_loss_series->update(epoch, current_loss);
            
            static float current_lr = 0.f;
            current_lr = lr;
            qDebug() << "lr: " << lr;
            m_lr_series->update(epoch, current_lr);

            qDebug() << "epoch: " << epoch;

            m_train_y_loss->setRange(0, current_loss * 1.2f);
            m_train_y_lr->setRange(0, current_lr * 1.2f);
        });

    m_update_timer->start(1000);
}

void gpu_performance_monitor::init_monitors() {
    Temp_Getter temp_getter = [](nvmlDevice_t device) {
        unsigned int temp = 0;
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        return temp;
    };
    
    Temp_Updater temp_updater = [this](unsigned int temp) {
        m_temp_status->setText(QString("Temperature: %1°C").arg(temp)); 
    };

    m_temp_monitor = std::make_unique<gpu_monitor_series< unsigned int, Temp_Getter, Temp_Updater>
                                                        > ( m_thermal_chart
                                                          , m_thermal_x_axis, m_y_temp
                                                          , "Temperature (°C)", Qt::red
                                                          , temp_getter, temp_updater
                                                          ) ;

    Power_Getter power_getter = [](nvmlDevice_t device) {
        unsigned int power_mwatts = 0;
        if (nvmlDeviceGetPowerUsage(device, &power_mwatts) == NVML_SUCCESS) {
            return static_cast<unsigned int>(power_mwatts / 1000);
        }
        return 0U;
    };

    Power_Updater power_updater = [this](unsigned int watts) {
        m_power_status->setText(QString("Power: %1 W").arg(watts));
    };

    m_power_monitor = std::make_unique<gpu_monitor_series< unsigned int, Power_Getter, Power_Updater>
                                                         > (m_thermal_chart
                                                           , m_thermal_x_axis, m_y_power
                                                           , "Power (W)", QColor(255, 165, 0)
                                                           , power_getter, power_updater
                                                           ) ;

    Mem_Getter mem_getter = [this](nvmlDevice_t device) -> double {
        nvmlMemory_t memory;
        if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
            return (memory.used * 100.0) / m_total_mem;
        }
        return 0.0;
    };

    Mem_Updater mem_updater = [this](double percent) {
        m_mem_status->setText(QString("Memory Usage: %1%").arg(percent, 0, 'f', 1));
    };

    m_mem_monitor = std::make_unique<gpu_monitor_series< double, Mem_Getter, Mem_Updater>
                                                       > (m_usage_chart
                                                         , m_usage_x_axis, m_y_mem
                                                         , "Memory Usage (%)", Qt::blue
                                                         , mem_getter, mem_updater
                                                         ) ;
                                            
    Util_Getter util_getter = [this](nvmlDevice_t device) {
        nvmlUtilization_t util;
        if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
            return util.gpu;
        }
        return 0U;
    };

    Util_Updater util_updater = [this](unsigned int percent) {
        m_gpu_util_status->setText(QString("GPU Usage: %1%").arg(percent));
    };

    m_gpu_util_monitor = std::make_unique<gpu_monitor_series< unsigned int, Util_Getter, Util_Updater>
                                                            > (m_usage_chart
                                                              , m_usage_x_axis, m_y_gpu_util
                                                              , "GPU Usage (%)", QColor(34, 139, 43)
                                                              , util_getter, util_updater
                                                              ) ;
                                        



    QLineSeries* mem_series = m_mem_monitor->series();
    mem_series->setVisible(true);

    connect(mem_series, &QLineSeries::hovered, [this](const QPointF& point, bool state) {
        if(state) {
            QToolTip::showText(QCursor::pos(), QString("时间: %1s\n显存占用: %2%").arg(point.x()) .arg(point.y(), 0, 'f', 1));
            m_mem_monitor->set_line_width(3);
        } else m_mem_monitor->set_line_width(1);
    });

    QLineSeries* util_series = m_gpu_util_monitor->series();
    util_series->setVisible(true);

    connect(util_series, &QLineSeries::hovered, [this](const QPointF& point, bool state) {
        if(state) {
            QToolTip::showText(QCursor::pos(), QString("时间: %1s\nGPU占用: %2%").arg(point.x()) .arg(point.y(), 0, 'f', 1));
            m_gpu_util_monitor->set_line_width(3);
        } else m_gpu_util_monitor->set_line_width(1);
    });

    QLineSeries* temp_series = m_temp_monitor->series();
    temp_series->setVisible(true);

    connect(temp_series, &QLineSeries::hovered, [this](const QPointF& point, bool state) {
        if(state) {
            QToolTip::showText(QCursor::pos(), QString("时间: %1s\n温度: %2°C").arg(point.x()) .arg(point.y(), 0, 'f', 1));
            m_temp_monitor->set_line_width(3);
        } else m_temp_monitor->set_line_width(1);
    });

    QLineSeries* power_series = m_power_monitor->series();
    power_series->setVisible(true);

    connect(power_series, &QLineSeries::hovered, [this](const QPointF& point, bool state) {
        if(state) {
            QToolTip::showText(QCursor::pos(), QString("时间: %1s\n功耗: %2W").arg(point.x()) .arg(point.y(), 0, 'f', 1));
            m_power_monitor->set_line_width(3);
        } else m_power_monitor->set_line_width(1);
    });
}

void gpu_performance_monitor::init_training_monitors() {
    Loss_Getter loss_getter = []() { return 0.f; };
    Loss_Updater loss_updater = [this](float) { };
    m_loss_series = std::make_unique<train_monitor_series< float, Loss_Getter, Loss_Updater>
                                                         > ( m_training_log_chart
                                                           , m_train_x_axis, m_train_y_loss
                                                           , "Loss", Qt::red
                                                           , loss_getter, loss_updater
                                                           ) ;
    
    Lr_Getter lr_getter = []() { return 0.f; };
    Lr_Updater lr_updater = [this](float) { };
    m_lr_series = std::make_unique<train_monitor_series< float, Lr_Getter, Lr_Updater>
                                                       > ( m_training_log_chart
                                                         , m_train_x_axis, m_train_y_lr
                                                         , "Lr", Qt::blue
                                                         , lr_getter, lr_updater
                                                         ) ;
}

void gpu_performance_monitor::data_received_from_socket() {

}

void gpu_performance_monitor::data_collection_thread() {
    QMutexLocker locker(&m_nvml_mutex);
    const qint64 current = QDateTime::currentSecsSinceEpoch() - m_start_time;
    
    m_temp_monitor->update(m_device, current);
    m_mem_monitor->update(m_device, current);
    m_gpu_util_monitor->update(m_device, current);
    m_power_monitor->update(m_device, current);

    QMetaObject::invokeMethod(this, [this, current]() {
        m_usage_x_axis->setRange(current - 60, current);
        m_thermal_x_axis->setRange(current - 60, current);
    });
}

bool gpu_performance_monitor::init_NVML() {
    QMutexLocker locker(&m_nvml_mutex);
    nvmlReturn_t result = nvmlInit_v2();
    if (result != NVML_SUCCESS) return false;

    result = nvmlDeviceGetHandleByIndex(0, &m_device);
    if (result != NVML_SUCCESS) return false;

    nvmlMemory_t memory;
    if (nvmlDeviceGetMemoryInfo(m_device, &memory) == NVML_SUCCESS) {
        m_total_mem = memory.total;
    }

    return true;
}


gpu_performance_monitor::~gpu_performance_monitor() {
    m_update_timer->stop();
    m_data_watcher.waitForFinished();
    if (m_nvml_initialized) nvmlShutdown();
}