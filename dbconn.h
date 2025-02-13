#ifndef DBCONN_H
#define DBCONN_H

#include <QSqlDatabase>
#include <QSqlError>
#include <qdatetime.h>
#include <qjsonobject.h>
#include <string_view>
#include <string>
#include <QSqlQuery>
#include <QVariant>


struct model_summary {
    QString train_id;
    QString save_path;
    QDateTime timestamp;
    double mp50;
    qint64 params;
};

class dbConn
{
public:
    static dbConn& instance(){
        static dbConn instance;
        return instance;
    }

    dbConn(const dbConn&) = delete;
    dbConn& operator=(const dbConn&) = delete;

    void setConnectionInfo(
        std::string_view host,
        int port,
        std::string_view dbName,
        std::string_view user,
        std::string_view pwd
        );


    int insertUserEdit(
        const QString& tableName,
        const QString& className,
        const QString& length,
        const QString& width,
        const QString& height,
        const QString& filePath,
        int size
        );


    bool
    openDatabase();

    void
    closeDatabase();

    QSqlDatabase
    getDatabase();

    std::pair<int, int>
    onItemNameChangedDb(const QString& className);

    int // 获取插入时 item 的 id
    get_inserted_item_id(const QSqlQuery& query) const;

    int // 根据 class_name 获取对应的 id
    get_inserted_item_class_id(const QString& class_name) const;

    int
    insertNewClassId(const QString& className);

    bool
    updateClassAmount(const int& classId);

    bool
    update_item_with_id( const QString& table_name
                       , const QString& id
                       , const QString& name
                       , const QString& length
                       , const QString& width
                       , const QString& height
                       );

    int 
    insert_new_trained_model(const QJsonObject& report);

    QList<model_summary> get_all_models();

private:
    dbConn() = default;
    ~dbConn() { closeDatabase(); }


    QSqlDatabase m_db;
    std::string m_host;
    int m_port{3306};
    std::string m_dbName;
    std::string m_user;
    std::string m_password;
};

#endif // DBCONN_H
