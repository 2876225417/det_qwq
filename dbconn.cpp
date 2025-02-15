#include "dbconn.h"
#include <QMessageBox>
#include <QDateTime>
#include <qjsonobject.h>
#include <qlineedit.h>
#include <qlogging.h>

void dbConn::setConnectionInfo( std::string_view host
                              , int port
                              , std::string_view dbName
                              , std::string_view user
                              , std::string_view password
                              ) {
    m_host = std::string(host);
    m_port = port;
    m_dbName = std::string(dbName);
    m_user = std::string(user);
    m_password = std::string(password);
}

bool dbConn::openDatabase(){
    m_db = QSqlDatabase::addDatabase("QPSQL");              // 使用 PostgreSQL
    m_db.setHostName(QString::fromStdString(m_host));       // 主机
    m_db.setPort(m_port);                                   // 端口
    m_db.setDatabaseName(QString::fromStdString(m_dbName)); // 数据库名
    m_db.setUserName(QString::fromStdString(m_user));       // 用户名
    m_db.setPassword(QString::fromStdString(m_password));   // 密码

    if(m_db.open()){
        qDebug() << "Database connection successful!";
        return true;
    } else {
        qDebug() << "Database connection failed: " << m_db.lastError().text();
        return false;
    }
}

void dbConn::closeDatabase(){
    if(m_db.isOpen()){
        m_db.close();
        qDebug() << "Database connection closed!";
    }
}

QSqlDatabase dbConn::getDatabase(){
    return m_db;
}

int
dbConn::insert_new_trained_model(const QJsonObject& report) {
    if (!dbConn::instance().getDatabase().isOpen()) {
        qDebug() << "Failed: Database not open!";
        return -1;
    }

    QSqlQuery query(dbConn::instance().getDatabase());

    query.prepare(
        "INSERT INTO training_records ("
        "train_id, save_path, timestamp, duration, params, layers,"
        "gflops, gradients, mAP50, mAP5095, precision, recall,"
        "preprocess_ms, inference_ms, postprocess_ms"
        ") VALUES ("
        ":train_id, :save_path, :timestamp, :duration, :params, :layers,"
        ":gflops, :gradients, :mAP50, :mAP5095, :precision, :recall,"
        ":preprocess_ms, :inference_ms, :postprocess_ms)"
    );

    QDateTime timestamp = QDateTime::fromString(report["timestamp"].toString(), Qt::ISODate);
    query.bindValue(":train_id", report["train_id"].toString());
    query.bindValue(":save_path", report["save_path"].toString());
    query.bindValue(":timestamp", timestamp);
    query.bindValue(":duration", report["duration"].toString());

    QJsonObject modelInfo = report["model_info"].toObject();
    query.bindValue(":params", modelInfo["params"].toVariant().toLongLong());
    query.bindValue(":layers", modelInfo["layers"].toInt());
    query.bindValue(":gflops", modelInfo["gflops"].toDouble());
    query.bindValue(":gradients", modelInfo["gradients"].toInt());

    QJsonObject valMetrics = report["val_metrics"].toObject();
    query.bindValue(":mAP50", valMetrics["mAP50"].toDouble());
    query.bindValue(":mAP5095", valMetrics["mAP5095"].toDouble());
    query.bindValue(":precision", valMetrics["precision"].toDouble());
    query.bindValue(":recall", valMetrics["recall"].toDouble());

    QJsonObject speedStats = report["speed_stats"].toObject();
    query.bindValue(":preprocess_ms", speedStats["preprocess"].toDouble());
    query.bindValue(":inference_ms", speedStats["inference"].toDouble());
    query.bindValue(":postprocess_ms", speedStats["postprocess"].toDouble());

    if (!query.exec()) {
        qCritical() << "Insert failed:" << query.lastError().text()
                   << "\nQuery:" << query.lastQuery();
        return -2;
    }
    return 1; // 成功返回1
}

QList<model_summary> dbConn::get_all_models() {
    QList<model_summary> models;

    if (!dbConn::instance().getDatabase().isOpen()) {
        qDebug() << "Failed: Database not open!";
        return models;
    }

    QSqlQuery query(dbConn::instance().getDatabase());
    query.prepare(
        "SELECT train_id, save_path, timestamp, mAP50, mAP5095, params, layers, precision, gradients "
        "FROM training_records "
        "ORDER BY timestamp DESC"  
    );
    if (!query.exec()) {
        qCritical() << "Query failed: " << query.lastError().text();
        return models;
    }

    while (query.next()) {
        model_summary model;
        model.train_id = query.value("train_id").toString();
        model.save_path = query.value("save_path").toString();
        model.timestamp = query.value("timestamp").toDateTime();
        model.map50 = query.value("mAP50").toDouble();
        model.map5095 = query.value("mAP5095").toDouble();
        model.params = query.value("params").toLongLong();
        model.layers = query.value("layers").toDouble();
        model.precision = query.value("precision").toDouble();
        model.gradients = query.value("gradients").toDouble();

        models.append(model);
    }

    qDebug() << "Fetch models' summary successfully: " << models.count();
    return models;
}

int
dbConn::insertUserEdit( const QString& tableName
                      , const QString& className
                      , const QString& length
                      , const QString& width
                      , const QString& height
                      , const QString& filePath
                      , int size
                      ) {
    if (!dbConn::instance().getDatabase().isOpen()) {
        qDebug() << "Failed: Database not open!";
        return -1;
    }

    QSqlQuery query(dbConn::instance().getDatabase());
    
    QString queryStr = QString("INSERT INTO %1 (class, length, width, height, filePath, latestAdd, size) "
                               "VALUES (:class, :length, :width, :height, :filePath, :latestAdd, :size)")
                           .arg(tableName);
                           
    query.prepare(queryStr);
    qDebug() << "className: " << className << ", length: " << length
             << ", width: " << width << ", height: " << height
             << ", filePath: " << filePath << ", size: " << size;

    query.bindValue(":class", className);
    query.bindValue(":length", length.toDouble());
    query.bindValue(":width", width.toDouble());
    query.bindValue(":height", height.toDouble());
    query.bindValue(":filePath", filePath);
    query.bindValue(":latestAdd", QDateTime::currentDateTime());
    query.bindValue(":size", size);

    if (!query.exec()) {
        qDebug() << "Insert failed: " << query.lastError().databaseText();
        return -1;
    }

    qDebug() << "Insert successful!";
    return get_inserted_item_id(query);
}

int
dbConn::get_inserted_item_id(const QSqlQuery& query) const {
    QVariant last_id = query.lastInsertId();
    if (last_id.isValid()){
        qDebug() << "Inserted row ID: " << last_id.toInt();
        return last_id.toInt();
    }
    else{
        qDebug() << "Failed to retrieve the inserted ID.";
        return -1;
    }
}

int
dbConn::get_inserted_item_class_id(const QString& class_name) const {
    QSqlQuery query;

    query.prepare("SELECT class_id FROM classes WHERE class = :class_name");

    query.bindValue(":class_name", class_name);

    if (!query.exec()) {
        qDebug() << "Failed to retrieve class_id: " << query.lastError().text();
        return -1;
    }

    if (query.next()) {
        return query.value("class_id").toInt();
    }

    return -1;
}


std::pair<int, int>
dbConn::onItemNameChangedDb(const QString& className){
    QSqlQuery query;
    query.prepare("SELECT class_id, amount FROM classes WHERE class = :class_name LIMIT 1");
    query.bindValue(":class_name", className);

    if(query.exec() && query.next()){
        int classId = query.value(0).toInt();
        int amount = query.value(1).toInt();
        return {classId, amount};
    } else {
        return {-1, 0};
    }
}


int     // 插入的新 class 的 id  
dbConn::insertNewClassId(const QString& className){

    QSqlQuery query;
    query.prepare("INSERT INTO classes (class, amount) VALUES (:class_name, 1)");
    query.bindValue(":class_name", className);
    qDebug() << "className" << className;
    if(!query.exec()){
        qDebug() << "Failed to insert new class: " << query.lastError().text();
        return -1;
    }

    return query.lastInsertId().toInt();
}

bool    // 更新 class 数量，以获取 class 的编号
dbConn::updateClassAmount(const int& classId){

    QSqlQuery query;
    query.prepare("UPDATE classes SET amount = amount + 1 WHERE class_id = :class_id");
    query.bindValue(":class_id", classId);
    qDebug() << "Here yes" << classId;
    if(!query.exec()){
        qDebug() << "Failed to update label amount";
        return false;
    }
    return true;
}

bool    // 通过 id 更新 item 信息
dbConn::update_item_with_id( const QString& table_name
                           , const QString& id
                           , const QString& name
                           , const QString& length
                           , const QString& width
                           , const QString& height
                           ) {
    if (!dbConn::instance().getDatabase().isOpen()) {
        qDebug() << "Failed: Database not open!";
        return false;
    }

    QSqlQuery query(dbConn::instance().getDatabase());

    QString query_str = QString("UPDATE %1 SET class = :name, length = :length, width = :width, height = :height WHERE id = :id")
                            .arg(table_name);

    query.prepare(query_str);

    query.bindValue(":name", name);
    query.bindValue(":length", length);
    query.bindValue(":width", width);
    query.bindValue(":height", height);
    query.bindValue(":id", id);

    if (!query.exec()) {
        qDebug() << "Update failed: " << query.lastError().text();
        return false;
    }

    qDebug() << "Update successful!";

    return true;
}





