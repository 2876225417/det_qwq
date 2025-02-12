

#ifndef GPU_PERFORMANCE_MONITOR_H
#define GPU_PERFORMANCE_MONITOR_H


#include <QWidget>
#include <QtCharts/QtCharts>
#include <QLabel>
#include <QTimer>

#include <QFutureWatcher>
#include <QtCharts/qvalueaxis.h>
#include <QtConcurrent/QtConcurrent>

#include <QtWebSockets/qwebsocket.h>
#include <deque>
#include <functional>
#include <memory>
#include <nvml.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qobject.h>
#include <qtmetamacros.h>




template <typename Data_Type,
          typename Data_Getter,
          typename Label_Updater>
class gpu_monitor_series {
public:
    gpu_monitor_series( QChart* chart
                      , QValueAxis* x_axis
                      , QValueAxis* y_axis
                      , const QString& name
                      , const QColor& color
                      , Data_Getter getter
                      , Label_Updater updater
                      , int max_poinst = 60
                      ) : m_getter{getter}
                        , m_updater{updater}
                        , m_series{new QLineSeries()}
                        , m_x_axis{x_axis}
                        , m_max_points{max_poinst}
                        {
        m_series->setName(name);
        m_series->setColor(color);
        chart->addSeries(m_series);
        m_series->attachAxis(x_axis);
        m_series->attachAxis(y_axis);
        y_axis->setTitleText(name);
    }

    void update(nvmlDevice_t device, qint64 timestamp) {
        Data_Type value = m_getter(device);
        m_updater(value);

        m_points.emplace_back(timestamp, value);
        
        if (m_points.size() > m_max_points) {
            m_points.pop_front();
        }

        if (m_points.size() % 5 == 0) {
            QVector<QPointF> q_points;
            q_points.reserve(m_points.size());
            for (const auto& p: m_points) {
                q_points.append(QPointF(p.first, p.second));
            }
            m_series->replace(q_points);
        } else {
            m_series->append(timestamp, value);
        }
    }

    void update_axis(qint64 current) {
        m_x_axis->setRange(current - m_max_points, current);
    }

    QLineSeries* series() const { return m_series; }
    void set_line_width(qreal width) {
        QPen pen = m_series->pen();
        pen.setWidth(width);
        m_series->setPen(pen);
    }

private:
    QLineSeries* m_series;
    QValueAxis* m_x_axis;
    Data_Getter m_getter;
    Label_Updater  m_updater;
    std::deque<std::pair<qint64, Data_Type>> m_points;
    int m_max_points;
};

#include <QObject>
#include <QtWebSockets/QWebSocket>
// @todo: 共享内存实现
class get_train_log_websocket: public QObject {
    Q_OBJECT
public:
    explicit get_train_log_websocket(QObject* parent)
        : QObject(parent) 
        {
            qDebug() << "websocket constructed!";
        connect (&m_websocket, &QWebSocket::connected
                , this
                , &get_train_log_websocket::on_connected
                ) ;
        connect (&m_websocket, &QWebSocket::textMessageReceived
                , this
                , &get_train_log_websocket::on_text_message_received
                ) ;
    }
    void connect_to_server(const QUrl& url) {
        m_url = url;
        m_websocket.open(url);
    }
signals:
    void new_data_received(float loss, float lr, int epoch);
private slots:
    void on_connected() {
        qDebug() << "Connected to training monitor server!";
        m_websocket.sendTextMessage("start_stream");
    }
    void on_text_message_received(QString message) {
        QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
        QJsonObject json = doc.object();

        emit new_data_received(
            json["loss"].toDouble(),
            json["lr"].toDouble(),
            json["epoch"].toInt());
    }
private:
    QWebSocket m_websocket;
    QUrl m_url;
};


class gpu_performance_monitor: public QWidget {
    Q_OBJECT
public:
    explicit gpu_performance_monitor(QWidget* parent = nullptr);
    ~gpu_performance_monitor();

private slots:
    // void handle_data_ready();
private: 
    bool init_NVML();
    void init_monitors();
    void init_training_monitors();
    void data_collection_thread();

    bool m_nvml_initialized = false;
    nvmlDevice_t m_device;
    unsigned long long m_total_mem = 0;

    QLabel* m_temp_status;
    QLabel* m_power_status;
    QLabel* m_mem_status;
    QLabel* m_gpu_util_status;

    QChart* m_thermal_chart;
    QChart* m_usage_chart;
    QChart* m_training_log_chart;
    QChartView* m_thermal_view;
    QChartView* m_usage_view;
    QChartView* m_training_log_view;
    QValueAxis* m_thermal_x_axis;
    QValueAxis* m_usage_x_axis;
    
    QValueAxis* m_train_x_axis_epoch;
    QValueAxis* m_train_y_loss;
    QValueAxis* m_train_y_lr;

    
    QValueAxis* m_y_temp;
    QValueAxis* m_y_power;
    QValueAxis* m_y_mem;
    QValueAxis* m_y_gpu_util;

    using Temp_Getter = std::function<unsigned int(nvmlDevice_t)>;
    using Temp_Updater = std::function<void(unsigned int)>;

    std::unique_ptr<gpu_monitor_series< unsigned int
                                      , Temp_Getter
                                      , Temp_Updater>
                                      > m_temp_monitor;

    using Mem_Getter = std::function<double(nvmlDevice_t)>;
    using Mem_Updater = std::function<void(double)>;

    std::unique_ptr<gpu_monitor_series< double
                                      , Mem_Getter
                                      , Mem_Updater>
                                      > m_mem_monitor;

    using Util_Getter = std::function<unsigned int(nvmlDevice_t)>;
    using Util_Updater = std::function<void(unsigned int)>;

    std::unique_ptr<gpu_monitor_series< unsigned int
                                      , Util_Getter
                                      , Util_Updater>
                                      > m_gpu_util_monitor;

    using Power_Getter = std::function<unsigned int(nvmlDevice_t)>;
    using Power_Updater = std::function<void(unsigned int)>;

    std::unique_ptr<gpu_monitor_series< unsigned int
                                      , Power_Getter
                                      , Power_Updater>
                                      > m_power_monitor;

    using Epoch_Getter = std::function<int()>;
    using Epoch_Setter = std::function<void(int)>;

    std::unique_ptr<gpu_monitor_series< int
                                      , Epoch_Getter
                                      , Epoch_Setter>
                                      > m_epoch_monitor; 

    using Lr_Getter = std::function<float()>;
    using lr_Setter = std::function<void(float)>;

    std::unique_ptr<gpu_monitor_series< float 
                                      , Epoch_Getter
                                      , Epoch_Setter>
                                      > m_lr_monitor;

    QTimer* m_update_timer;
    QFutureWatcher<void> m_data_watcher;
    qint64 m_start_time;
    QMutex m_nvml_mutex;

    get_train_log_websocket* m_train_log_websocket;
};
#endif