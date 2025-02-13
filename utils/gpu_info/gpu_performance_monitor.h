

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
#include <qoverload.h>
#include <qprogressbar.h>
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

template <typename Data_Type,
          typename Data_Getter,
          typename Label_Updater>
class train_monitor_series {
public:
    train_monitor_series( QChart* chart
                      , QValueAxis* x_axis
                      , QValueAxis* y_axis
                      , const QString& name
                      , const QColor& color
                      , Data_Getter getter
                      , Label_Updater updater
                      , int max_poinst = 100
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

    void update(int epoch, Data_Type value) {
        m_updater(value);
        m_points.emplace_back(epoch, value);
        
        if (m_points.size() > m_max_points) {
            m_points.pop_front();
        }

        QVector<QPointF> q_points;
        q_points.reserve(m_points.size());
        for (const auto& p: m_points) {
            q_points.append(QPointF(p.first, p.second));
        }
        m_series->replace(q_points);
        
        m_x_axis->setRange(epoch - m_max_points, epoch);        
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
        : QObject{parent}
        , m_reconnect_timer{new QTimer(this)}
        {
            m_reconnect_timer->setInterval(3000);
            m_reconnect_timer->setSingleShot(true);

            qDebug() << "websocket constructed!";
        connect ( &m_websocket, &QWebSocket::connected
                , this
                , &get_train_log_websocket::on_connected
                ) ;
        connect ( &m_websocket, &QWebSocket::textMessageReceived
                , this
                , &get_train_log_websocket::on_text_message_received
                ) ;
        connect ( &m_websocket
                , &QWebSocket::disconnected
                , this
                , &get_train_log_websocket::start_reconnect
                ) ;
        connect ( &m_websocket
                , QOverload<QAbstractSocket::SocketError>::of(&QWebSocket::errorOccurred)
                , this
                , [this](QAbstractSocket::SocketError error) {
                    qDebug() << "WebSocket error: " << error << m_websocket.errorString();
                    start_reconnect();
                });
        connect ( m_reconnect_timer
                , &QTimer::timeout
                , this
                , &get_train_log_websocket::do_connect
                ) ;
    }
    void connect_to_server(const QUrl& url) {
        m_url = url;
        m_websocket.open(url);
    }
signals:
    void new_data_received(float loss, float lr, int epoch);
    void train_started(QString train_id, int total_batches, int total_epochs);
    void train_ended(QJsonObject summary);
    void train_progress(int current_epoch, float progress);
    void training_metrics(QJsonObject metrics);

public slots:
    void do_connect() {
        if (m_websocket.state() != QAbstractSocket::UnconnectedState) {
            qDebug() << "Already connected, current state: " << m_websocket.state();
            return;
        }
        qDebug() << "Connecting to"  << m_url.toString();
        m_websocket.open(m_url);
    }
    void start_reconnect() {
        //int interval = qMin(1000 * (1 << m_reconnect_attempts), MAX_RETRY_INTERVAL);
        // m_reconnect_timer->setInterval(interval);
        // m_reconnect_attempts++;

        if (!m_reconnect_timer->isActive())
            //qDebug() << "Attempt to reconnect in :" << interval / 1000;
            m_reconnect_timer->start();
    }
private slots:
    void on_connected() {
        qDebug() << "Connected to training monitor server!";
        m_websocket.sendTextMessage("start_stream");
    }
    void on_text_message_received(QString message) {
        QJsonParseError parse_error;
        QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8(), &parse_error);

        if (!doc.isObject() || parse_error.error != QJsonParseError::NoError) {
            qWarning() << "Failed to parse Json: " << parse_error.errorString();
            return;
        }

        if (!doc.isObject()) {
            qWarning() << "Received message is not Json object: " << message;
            return;
        }

        QJsonObject json = doc.object();
        QString event_type = json.value("event_type").toString();

        if (event_type == "train_start") handle_train_start_event(json);
        else if(event_type == "train_end") handle_train_end_event(json);
        else if(event_type == "epoch_end") handle_epoch_end_event(json);
        else qWarning() << "Unknown event type: " << event_type;
    }
private:
    void handle_train_start_event(const QJsonObject& json) {
        QString train_id = json.value("train_id").toString();
        int total_batches = json.value("total_batches").toInt();
        int total_epochs = json.value("total_epochs").toInt(-1);

        if (train_id.isEmpty() || total_epochs <= 0) {
            qWarning() << "Invalid start training metrics: " << json;
            return;
        }

        emit train_started(train_id, total_batches, total_epochs);
    }

    void handle_train_end_event(const QJsonObject& json) {
        QString train_id = json.value("train_id").toString();
        if (train_id.isEmpty()) {
            qWarning() << "Invalid end training metrics: " << json;
            return;
        }
        emit train_ended(json);
    }

    void handle_epoch_end_event(const QJsonObject& json) {
        bool ok = true;
        int epoch = json.value("epoch").toInt();
        if ( !ok || epoch < 0) {
            qWarning() << "Invalid epoch data: " << json;
            return;
        }

        emit training_metrics(json);

        QString progress_str = json.value("progress").toString().replace("%", "");
        float progress = progress_str.toFloat();
        if (ok) emit train_progress(epoch, progress);
    }



    QWebSocket m_websocket;
    QUrl m_url;
    QTimer* m_reconnect_timer; 
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
    void data_received_from_socket();

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
    
    QValueAxis* m_train_x_axis;
    QValueAxis* m_train_y_loss;
    QValueAxis* m_train_y_lr;
    QProgressBar* m_training_progress;

    
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


    using Loss_Getter = std::function<float()>;
    using Loss_Updater = std::function<void(float)>;

    std::unique_ptr<train_monitor_series< float
                                        , Loss_Getter
                                        , Loss_Updater>
                                        > m_loss_series;

    using Lr_Getter = std::function<float()>;
    using Lr_Updater = std::function<void(float)>;

    std::unique_ptr<train_monitor_series< float 
                                        , Lr_Getter
                                        , Lr_Updater>
                                        > m_lr_series;

    QTimer* m_update_timer;
    QFutureWatcher<void> m_data_watcher;
    qint64 m_start_time;
    QMutex m_nvml_mutex;
public:
    get_train_log_websocket* m_train_log_websocket;
};
#endif