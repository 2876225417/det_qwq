#include "sidebar.h"

#include <QIcon>
#include <QDebug>


Sidebar::Sidebar(QWidget *parent)
    : QWidget{parent}
{
    listWidget = new QListWidget(this);
    listWidget->setIconSize(QSize(40, 40));
    listWidget->setStyleSheet("QListWidget { background-color: #2e3a44; color: white; }");

    listWidget->setFrameShape(QFrame::NoFrame);

    /* Sub Page Items
     *
     *
     *
     *
     *
     */

    QListWidgetItem* Dashboard = new QListWidgetItem(QIcon(":icons/icons/dashboard.svg"), "控制面板");
    listWidget->addItem(Dashboard);

    QListWidgetItem* Train = new QListWidgetItem(QIcon(":icons/icons/robot.svg"),"模型训练");
    listWidget->addItem(Train);

    QListWidgetItem* DataBase = new QListWidgetItem(QIcon(":icons/icons/database.svg"), "数据库");
    listWidget->addItem(DataBase);

    // QListWidgetItem* DataSet = new QListWidgetItem("数据集");
    // listWidget->addItem(DataSet);


    QListWidgetItem* Configuration = new QListWidgetItem(QIcon(":icons/icons/cog.svg"), "配置");
    listWidget->addItem(Configuration);

    QListWidgetItem* _detection = new QListWidgetItem(QIcon(":icons/icons/detect.svg"), "模型检测");
    listWidget->addItem(_detection);

    QListWidgetItem* _serial_port_configuration = new QListWidgetItem(QIcon(":/icons/icons/serial-port.svg"), "端口配置");
    listWidget->addItem(_serial_port_configuration);

    QListWidgetItem* opengl = new QListWidgetItem("opengl");
    listWidget->addItem(opengl);



    connect(listWidget, &QListWidget::itemClicked, [this](QListWidgetItem* item){
        emit itemClicked(listWidget->row(item));
    });

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(listWidget);
    setLayout(layout);
}


void Sidebar::onItemClicked(QListWidgetItem* item){
    int index = listWidget->row(item);
    emit itemClicked(index);
}
