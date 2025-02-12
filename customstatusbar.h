#ifndef CUSTOMSTATUSBAR_H
#define CUSTOMSTATUSBAR_H




#include <QStatusBar>
#include <QWidget>
#include <QLabel>

class CustomStatusBar : public QStatusBar
{
    Q_OBJECT
public:
    explicit CustomStatusBar(QWidget* parent = nullptr);

    void showStatusMessage(
        const QString& message,
        int timeout = 0
        );

    void addCustomWidget(QWidget* widget);

private:
    QLabel* statusLabel;
};

#endif // CUSTOMSTATUSBAR_H
