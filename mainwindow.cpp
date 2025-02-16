#include "mainwindow.h"
#include <QLabel>
#include <QStackedWidget>
#include <QWidget>
#include <QVBoxLayout>
#include <QSplitter>
#include <QHBoxLayout>
#include <qaction.h>
#include <qtoolbar.h>



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{


    sidebar = new Sidebar(this);

    stackedWidget = new QStackedWidget(this);

    homePage = new QWidget();



    _dashboard = new Dashboard();

    _train = new TrainBoard();

    _database = new Database();

    _configuration = new Configuration();

    // _openglboard = new OpenGLBoard();

    _sp_config_board = new serial_port_configuration_board();

    _detection_board = new detectionBoard();

    stackedWidget->addWidget(_dashboard);
    stackedWidget->addWidget(_train);
    stackedWidget->addWidget(_database);
    stackedWidget->addWidget(_configuration);
    stackedWidget->addWidget(_detection_board);
    stackedWidget->addWidget(_sp_config_board);
    // stackedWidget->addWidget(_openglboard);

    QSplitter* splitter = new QSplitter(Qt::Horizontal);
    splitter->addWidget(sidebar);
    splitter->addWidget(stackedWidget);

    sidebar->setFixedWidth(150);
    splitter->setSizes(QList<int>() << 200 << 800);
    setCentralWidget(splitter);

    connect(sidebar, &Sidebar::itemClicked, this, &MainWindow::onSidebarItemClicked);

    setWindowTitle("基于Qt的物体检测平台");
    this->resize(1544, 1009);

    customStatusBar = new CustomStatusBar(this);
    setStatusBar(customStatusBar);
    // QToolBar* toolbar = new QToolBar("Toolbar");
    // QAction* start = toolbar->addAction("start");
    // addToolBar(toolbar);

    connect(_database, &Database::connectionStatusChange, this, &MainWindow::updateStatusBar);
    // QLabel* addtionalInfo = new QLabel("ready", this);
    // customStatusBar->addCustomWidget(addtionalInfo);
}

MainWindow::~MainWindow() {}

void MainWindow::updateStatusBar(const QString& statusMessage){
    customStatusBar->showStatusMessage(statusMessage);
}


void MainWindow::onSidebarItemClicked(int index){
    qDebug() << "Sidebar item clicked: " << index;

    stackedWidget->setCurrentIndex(index);

}
