#ifndef SIDEBAR_H
#define SIDEBAR_H

#include <QWidget>
#include <QListWidget>
#include <QListWidgetItem>
#include <QVBoxLayout>


class Sidebar : public QWidget
{
    Q_OBJECT
public:
    explicit Sidebar(QWidget *parent = nullptr);

signals:
    void itemClicked(int index);

private slots:
    void onItemClicked(QListWidgetItem* item);

private:
    QListWidget* listWidget;


    void setupUI();

    void loadQss();

};

#endif // SIDEBAR_H
