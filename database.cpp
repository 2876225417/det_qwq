#include "database.h"
#include <qevent.h>


Database::Database(QWidget *parent):
    QWidget{parent}
{
    statusLabel = new QLabel("Status: Not Connected", this);

    hostEdit = new QLineEdit(this);
    portEdit = new QLineEdit(this);
    dbNameEdit = new QLineEdit(this);
    userEdit = new QLineEdit(this);
    passwordEdit = new QLineEdit(this);
    passwordEdit->setEchoMode(QLineEdit::Password);

    savePasswordCheckBox = new QCheckBox("save password", this);

    connectButton = new QPushButton("Connect to Database", this);
    showDataButton = new QPushButton("Show data", this);
    queryEdit = new QLineEdit(this);

    tableNameEdit = new QLineEdit(this);
    tableNameEdit->setPlaceholderText("Enter table name");

    QFormLayout* formLayout = new QFormLayout;
    formLayout->addRow("Host: ", hostEdit);
    formLayout->addRow("Port: ", portEdit);
    formLayout->addRow("Database Name: ", dbNameEdit);
    formLayout->addRow("Username: ", userEdit);
    formLayout->addRow("Password: ", passwordEdit);
    formLayout->addRow("", savePasswordCheckBox);
    formLayout->addRow("SQL query: ", queryEdit);
    formLayout->addRow("Table Name: ", tableNameEdit);

    classTableWidget = new QTableWidget(this);
    classTableWidget->setColumnCount(2);
    classTableWidget->setHorizontalHeaderLabels(QStringList() << "Class" << "Amount");

    tableWidgetQueried =  new QTableWidget(this);

    QVBoxLayout* leftLayout = new QVBoxLayout;
    leftLayout->addLayout(formLayout);
    leftLayout->addWidget(connectButton);
    leftLayout->addWidget(statusLabel);
    leftLayout->addWidget(showDataButton);
    leftLayout->addWidget(classTableWidget);

    tableView = new QTableView(this);
    tableView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    imageLabel = new QLabel(this);
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    QWidget* imageContainer = new QWidget(this);
    QVBoxLayout* imageLayout = new QVBoxLayout(imageContainer);
    imageLayout->addWidget(imageLabel);

    imageContainer->setStyleSheet("background-color: #444;"
                                  "border-radius: 5px");


    QHBoxLayout* mainLayout = new QHBoxLayout(this);
    mainLayout->addLayout(leftLayout, 1);

    QVBoxLayout* rightLayout = new QVBoxLayout;
    QHBoxLayout* bottomLayout = new QHBoxLayout;

    bottomLayout->addWidget(tableWidgetQueried);

    bottomLayout->addWidget(imageContainer);

    rightLayout->addWidget(tableView);
    rightLayout->addLayout(bottomLayout);
    mainLayout->addLayout(rightLayout, 3);

    loadSettings();

    connect(connectButton, &QPushButton::clicked, this, &Database::onConnectButtonClicked);
    connect(showDataButton, &QPushButton::clicked, this, &Database::onShowDataButtonClicked);

    classTableWidget->setSelectionMode(QAbstractItemView::SingleSelection);
    connect(classTableWidget, &QTableWidget::itemClicked, this, &Database::onClassTableItemClicked);

    connect(tableWidgetQueried, &QTableWidget::cellClicked, this, &Database::onFilePathCellClicked);
}

void Database::onConnectButtonClicked(){
    QString host = hostEdit->text();
    int port = portEdit->text().toInt();
    QString dbName = dbNameEdit->text();
    QString user = userEdit->text();
    QString password = passwordEdit->text();

    dbConn::instance().setConnectionInfo(host.toStdString(),
                               port,
                               dbName.toStdString(),
                               user.toStdString(),
                               password.toStdString());

    if(dbConn::instance().openDatabase()){
        QString statusMessage = QString("Status: connected to %1:%2 as %3")
                                    .arg(host)
                                    .arg(port)
                                    .arg(user);

        emit connectionStatusChange(statusMessage);

        statusLabel->setText("Status: Connection successful!");
        QMessageBox::information(this, "Success", "Database connection successful!");

        saveSettings();
    } else {
        QString statusMessage = "Status: Connection failed";
        emit connectionStatusChange(statusMessage);
        statusLabel->setText("Stasu: Connection connection failed!");
        QMessageBox::information(this, "Error", "Database Connection failed");
    }
}


void Database::onFilePathCellClicked(int row, int column){
    if(column == 5){
        QString filePath = tableWidgetQueried->item(row, column)->text();
        if(!filePath.isEmpty() && QFile::exists(filePath)){
            QPixmap pixmap(filePath);
            if(!pixmap.isNull()){
                imageLabel->setPixmap(pixmap.scaled(imageLabel->size(), Qt::KeepAspectRatio));
            } else {
                QMessageBox::warning(this, "warning", "Fail to load image.");
            }
        } else {
            QMessageBox::warning(this, "Error", "Invalid file path.");
        }
    }
}


void Database::onShowDataButtonClicked(){
    if(!dbConn::instance().getDatabase().isOpen()){
        QMessageBox::warning(this, "Error", "Please connect to the database first");
        return;
    }

    QString tableName = tableNameEdit->text();

    if(tableName.isEmpty()){
        QMessageBox::warning(this, "Error", "Please enter a table name");
        return;
    }

    model = new QSqlTableModel(this, dbConn::instance().getDatabase());
    model->setTable(tableName);
    model->select();

    tableView->setModel(model);
    tableView->resizeColumnsToContents();

    QSqlQuery query(dbConn::instance().getDatabase());
    query.prepare("SELECT class, COUNT(*) as amount from " + tableName + " GROUP BY class");

    if(!query.exec()){
        QMessageBox::warning(this, "Error", "Failed to fetch class data: " + query.lastError().text());
        return;
    }

    classTableWidget->setRowCount(0);

    int row = 0;

    while(query.next()){
        classTableWidget->insertRow(row);
        classTableWidget->setItem(row, 0, new QTableWidgetItem(query.value(0).toString()));
        classTableWidget->setItem(row, 1, new QTableWidgetItem(query.value(1).toString()));
        row++;
    }
}

void Database::onClassTableItemClicked(QTableWidgetItem* item){
    QString clickedClass = item->text();
    showClassData(clickedClass);
}

void Database::contextMenuEvent(const QPoint& pos){
    QModelIndex index = classTableWidget->indexAt(pos);

    if(index.isValid()){
        qDebug() << "clicked";

        QMenu contextMenu(this);

        QString clickedClass = classTableWidget->item(index.row(), 0)->text();

        QAction* actionShowClassData = new QAction("show all data for " + clickedClass, &contextMenu);

        connect(actionShowClassData, &QAction::triggered, this, [this, clickedClass]() {
            showClassData(clickedClass);
        });

        contextMenu.addAction(actionShowClassData);

        contextMenu.exec(classTableWidget->mapToGlobal(pos));

    }
}

void Database::showClassData(const QString& className){
    QString tableName = tableNameEdit->text();

    if(!dbConn::instance().getDatabase().isOpen()){
        QMessageBox::warning(this, "Error", "Please connect to the database first");
        return;
    }

    QSqlQuery query(dbConn::instance().getDatabase());
    query.prepare("SELECT * FROM " + tableName + " WHERE class = :className");
    query.bindValue(":className", className);


    if(!query.exec()){
        QMessageBox::warning(this, "Error", "Failed to fetch data for class" + className + ": " + query.lastError().text());
        return;
    }

    tableWidgetQueried->clearContents();
    tableWidgetQueried->setRowCount(0);

    int columnCount = query.record().count();
    tableWidgetQueried->setColumnCount(columnCount);

    for(int i = 0; i < columnCount; ++i){
        tableWidgetQueried->setHorizontalHeaderItem(i, new QTableWidgetItem(query.record().fieldName(i)));
    }

    int row = 0;
    while(query.next()){
        tableWidgetQueried->insertRow(row);
        for(int col = 0; col < columnCount; ++col){
            tableWidgetQueried->setItem(row, col, new QTableWidgetItem(query.value(col).toString()));
        }
        ++row;
    }

    tableView->resizeColumnsToContents();
}

void Database::saveSettings(){
    QSettings settings("qwq", "database");

    settings.setValue("host", hostEdit->text());
    settings.setValue("port", portEdit->text());
    settings.setValue("dbName", dbNameEdit->text());
    settings.setValue("username", userEdit->text());

    if(savePasswordCheckBox->isChecked()){
        settings.setValue("password", passwordEdit->text());
    } else {
        settings.remove("password");
    }

    settings.setValue("tableName", tableNameEdit->text());

}


void Database::loadSettings(){
    QSettings settings("qwq", "database");

    hostEdit->setText(settings.value("host", "localhost").toString());
    portEdit->setText(settings.value("port", 3306).toString());
    dbNameEdit->setText(settings.value("dbName", "dataset").toString());
    userEdit->setText(settings.value("username", "root").toString());

    if(settings.contains("password")){
        passwordEdit->setText(settings.value("password").toString());
        savePasswordCheckBox->setChecked(true);
    } else {
        passwordEdit->clear();
        savePasswordCheckBox->setChecked(false);
    }

    tableNameEdit->setText(settings.value("tableName").toString());

}






