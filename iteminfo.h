#ifndef ITEMINFO_H
#define ITEMINFO_H

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QPixmap>
#include <QPaintEvent>
#include <QPainter>
#include <QLineEdit>
#include <QPushButton>
#include <QMessageBox>

#include "dbconn.h"

// gallery 中的单个 卡片组件
class ItemInfo: public QWidget
{
    Q_OBJECT
public:
    explicit ItemInfo(
        const QPixmap& image = QPixmap(),
        const QString& id = "",
        const QString& name = "",
        const QString& length = "",
        const QString& width = "",
        const QString& height = "",
        QWidget* parent = nullptr
        );

    void set_id(const QString& id);
    void setName(const QString& name);
    void setLength(const QString& length);
    void setWidth(const QString& width);
    void setHeight(const QString& height);
    void setImage(const QPixmap& image);

    QString get_item_id() const;

    QString get_id() const;
    QString getName() const;
    QString getLength() const;
    QString getWidth() const;
    QString getHeight() const;
    QPixmap getImage() const;

private:
    QLabel* imageLabel;

    QLabel* id_label;
    QLineEdit* id_edit;

    QLabel* nameLabel;
    QLineEdit* nameEdit;

    QLabel* lengthLabel;
    QLineEdit* lengthEdit;

    QLabel* widthLabel;
    QLineEdit* widthEdit;

    QLabel* heightLabel;
    QLineEdit* heightEdit;

    QVBoxLayout* InfoLayout;
    QHBoxLayout* InfoBlock;


    void paintEvent(QPaintEvent* event) override;

private slots:
    void onEditButtonClicked();

};

#endif // ITEMINFO_H
