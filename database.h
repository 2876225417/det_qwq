#ifndef DATABASE_H
#define DATABASE_H

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QFormLayout>
#include <QSqlDatabase>
#include <QSqlError>
#include <QDebug>
#include <QVBoxLayout>
#include <QSqlError>
#include <QMessageBox>
#include <QSqlTableModel>
#include <QTableView>
#include <QSqlQueryModel>
#include <QSettings>
#include <QHBoxLayout>
#include <QComboBox>
#include <QCheckBox>
#include <QSqlQuery>
#include <QTableWidget>
#include <QMenu>
#include <QAction>
#include <QTableWidgetItem>
#include <QSqlRecord>
#include <QFile>
#include <QDateTime>

#include "dbconn.h"


class Database : public QWidget
{
    Q_OBJECT
public:
    explicit Database(QWidget *parent = nullptr);

    bool insertUserEdit(
        const QString& tableName,
        const QString& className,
        const QString& length,
        const QString& width,
        const QString& height,
        const QString& filePath,
        int size
        );

signals:
    void connectionStatusChange(const QString& statusMessage);

private slots:
    void onConnectButtonClicked();
    void onShowDataButtonClicked();
    void onClassTableItemClicked(QTableWidgetItem* item);
    void onFilePathCellClicked(int row, int column);


    void showClassData(const QString& className);

protected:
    void contextMenuEvent(const QPoint& pos);

private:
    QLabel* statusLabel;
    QLineEdit* hostEdit;
    QLineEdit* portEdit;
    QLineEdit* dbNameEdit;
    QLineEdit* userEdit;
    QLineEdit* passwordEdit;
    QCheckBox* savePasswordCheckBox;
    QPushButton* connectButton;

    QPushButton* showDataButton;
    QLineEdit* queryEdit;
    QLineEdit* tableNameEdit;

    QLabel* imageLabel;
    QTableWidget* classTableWidget;
    QTableWidget* tableWidgetQueried;


    QSqlTableModel* model;
    QTableView* tableView;


    void saveSettings();
    void loadSettings();

};

#endif // DATABASE_H
