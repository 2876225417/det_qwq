

// #include <QCoreApplication>
// #include <QSqlDatabase>
// #include <QSqlQuery>
// #include <QSqlError>
// #include <QDebug>


// int main(int argc, char *argv[])
// {
//     QCoreApplication a(argc, argv);

//     // 创建数据库连接
//     QSqlDatabase db = QSqlDatabase::addDatabase("QPSQL");  // 使用 PostgreSQL 驱动

//     // 设置连接参数
//     db.setHostName("localhost");  // 数据库主机
//     db.setPort(5432);             // PostgreSQL 默认端口
//     db.setDatabaseName("postgres"); // 数据库名
//     db.setUserName("postgres"); // 用户名
//     db.setPassword("20041025"); // 密码

//     // 打开数据库连接
//     if (!db.open()) {
//         qDebug() << "连接数据库失败:" << db.lastError().text();
//         return -1;
//     }

//     qDebug() << "数据库连接成功!";

//     // 执行 SQL 查询
//     QSqlQuery query;
//     if (query.exec("SELECT * FROM your_table")) {
//         while (query.next()) {
//             QString result = query.value(0).toString();  // 获取查询结果（第一列）
//             qDebug() << result;
//         }
//     } else {
//         qDebug() << "查询失败:" << query.lastError().text();
//     }

//     return a.exec();
// }


#include <QApplication>
#include <QMainWindow>
#include <iostream>

#include "mainwindow.h"

// #include "onnxruntime_inference_session.h"

#include <QSerialPort>
#include <QSerialPortInfo>
#include <QDebug>

#include <QProcess>
#include <qapplication.h>
#include <QStyleFactory>
#include <qnamespace.h>
#include <qsurfaceformat.h>

int main(int argc, char* argv[]){
    // QCoreApplication::setAttribute(Qt::AA_UseOpenGLES, false);
    // QCoreApplication::setAttribute(Qt::AA_UseSoftwareOpenGL, false);
    // QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL);
    
    // QSurfaceFormat fmt;
    // fmt.setRenderableType(QSurfaceFormat::OpenGL);
    // fmt.setVersion(3, 3);
    // fmt.setProfile(QSurfaceFormat::CoreProfile);
    // fmt.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    // fmt.setOption(QSurfaceFormat::DebugContext);
    // fmt.setRedBufferSize(8);
    // fmt.setGreenBufferSize(8);
    // fmt.setBlueBufferSize(8);
    // fmt.setAlphaBufferSize(0);
    // fmt.setDepthBufferSize(24);
    // fmt.setStencilBufferSize(8);
    // QSurfaceFormat::setDefaultFormat(fmt);
    QApplication app(argc, argv);
    // QApplication::setStyle(QStyleFactory::create("Fusion"));
    MainWindow m;
    m.show();

    return app.exec();
}

// #include <iostream>
// #include <vector>

// // 声明 CUDA 函数
// extern "C" void highLoad(int* result, int N);

// int main() {
//     const int N = 1024;  // 假设我们使用 1024 个线程来生成负载
//     std::vector<int> result(N, 0);  // 结果向量，初始化为 0

//     // 调用高负载函数
//     highLoad(result.data(), N);

//     // 打印部分计算结果，检查是否正常
//     std::cout << "Result[0] = " << result[0] << std::endl;
//     std::cout << "Result[N-1] = " << result[N - 1] << std::endl;

//     // 打印部分计算结果（例如第一行）
//     std::cout << "First few results: ";
//     for (int i = 0; i < 10; ++i) {
//         std::cout << result[i] << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }