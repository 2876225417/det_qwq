#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QStackedWidget>
#include <QSplitter>
#include <QStatusBar>


/* Components Import */
#include "sidebar.h"
#include "customstatusbar.h"

/* Sub Pages Import */
#include "dashboard.h"
#include "database.h"
#include "trainboard.h"
#include "configuration.h"
// #include "openglboard.h"
#include "detectionboard.h"
#include "serial_port_configuration_board.h"




class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onSidebarItemClicked(int index);
    void updateStatusBar(const QString& statusMessage);

private:

    QStackedWidget* stackedWidget;
    QWidget* homePage;
    QWidget* settingsPage;

    /* Components Member */
    Sidebar* sidebar;
    CustomStatusBar* customStatusBar;

    /* Sub Pages Member */
    Dashboard* _dashboard;
    Database* _database;
    TrainBoard* _train;
    Configuration* _configuration;
    detectionBoard* _detection_board;
    serial_port_configuration_board* _sp_config_board;

    // OpenGLBoard* _openglboard;



};
#endif // MAINWINDOW_H
