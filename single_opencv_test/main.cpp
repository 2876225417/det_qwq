#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QTimer>
#include <QThread>
#include <opencv2/opencv.hpp>
#include <QImage>

class VideoThread : public QThread {
    Q_OBJECT
public:
    VideoThread(QObject *parent = nullptr) : QThread(parent), cap(0) {}

protected:
    void run() override {
        while (true) {
            cv::Mat frame;
            cap >> frame; // 捕获视频帧

            if (frame.empty()) break;

            // 转换为QImage格式
            QImage img(frame.data, frame.cols, frame.rows, frame.step, QImage::Format_RGB888);
            emit frameCaptured(img); // 发射信号，将捕获到的帧传递给主线程
            msleep(5); // 暂停30ms，控制帧率
        }
    }

signals:
    void frameCaptured(const QImage &img);

private:
    cv::VideoCapture cap;
};

class VideoWidget : public QWidget {
    Q_OBJECT
public:
    VideoWidget(QWidget *parent = nullptr) : QWidget(parent) {
        QVBoxLayout *layout = new QVBoxLayout(this);
        label = new QLabel(this);
        layout->addWidget(label);

        // 创建视频线程
        videoThread = new VideoThread(this);
        connect(videoThread, &VideoThread::frameCaptured, this, &VideoWidget::updateFrame);
        videoThread->start();
    }

private slots:
    void updateFrame(const QImage &img) {
        label->setPixmap(QPixmap::fromImage(img));
    }

private:
    QLabel *label;
    VideoThread *videoThread;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    VideoWidget widget;
    widget.show();

    return app.exec();
}

#include "main.moc"
