#include "iteminfo.h"

ItemInfo::ItemInfo( const QPixmap& image
                  , const QString& id
                  , const QString& name
                  , const QString& length
                  , const QString& width
                  , const QString& height
                  , QWidget* parent
                  ): QWidget(parent) {
    imageLabel = new QLabel(this);
    imageLabel->setPixmap(image.scaled(150, 150, Qt::KeepAspectRatio));
    imageLabel->setFixedSize(150, 150);

    id_label = new QLabel("ID: ", this);
    id_label->setFixedWidth(80);
    id_edit = new QLineEdit(id, this);
    id_edit->setReadOnly(true);


    nameLabel = new QLabel("Name: ", this);
    nameLabel->setFixedWidth(80);
    nameEdit = new QLineEdit(name, this);
    nameEdit->setReadOnly(true);

    lengthLabel = new QLabel("Length: ", this);
    lengthLabel->setFixedWidth(80);
    lengthEdit = new QLineEdit(length, this);
    lengthEdit->setReadOnly(true);

    widthLabel = new QLabel("Width: ", this);
    widthLabel->setFixedWidth(80);
    widthEdit = new QLineEdit(width, this);
    widthEdit->setReadOnly(true);

    heightLabel = new QLabel("Height: ", this);
    heightLabel->setFixedWidth(80);
    heightEdit = new QLineEdit(height, this);
    heightEdit->setReadOnly(true);

    InfoBlock = new QHBoxLayout(this);
    InfoLayout = new QVBoxLayout(this);

    QHBoxLayout* id_layout = new QHBoxLayout();
    id_layout->addWidget(id_label);
    id_layout->addWidget(id_edit);

    QHBoxLayout* nameLayout = new QHBoxLayout();
    nameLayout->addWidget(nameLabel);
    nameLayout->addWidget(nameEdit);

    QHBoxLayout* lengthLayout = new QHBoxLayout();
    lengthLayout->addWidget(lengthLabel);
    lengthLayout->addWidget(lengthEdit);

    QHBoxLayout* widthLayout = new QHBoxLayout();
    widthLayout->addWidget(widthLabel);
    widthLayout->addWidget(widthEdit);

    QHBoxLayout* heightLayout = new QHBoxLayout();
    heightLayout->addWidget(heightLabel);
    heightLayout->addWidget(heightEdit);

    InfoLayout->addLayout(id_layout);
    InfoLayout->addLayout(nameLayout);
    InfoLayout->addLayout(lengthLayout);
    InfoLayout->addLayout(widthLayout);
    InfoLayout->addLayout(heightLayout);


    InfoBlock->addWidget(imageLabel);
    InfoBlock->addLayout(InfoLayout);

    setLayout(InfoBlock);
    setFixedHeight(200);
    setMaximumWidth(400);
    setMaximumHeight(160);

    QPushButton* editButton = new QPushButton("Edit", this);
    editButton->setFixedSize(80, 30);
    InfoLayout->addWidget(editButton);

    connect(editButton, &QPushButton::clicked, this, &ItemInfo::onEditButtonClicked);
}

// 信息卡片的外观
void ItemInfo::paintEvent(QPaintEvent* event){
    QWidget::paintEvent(event);
    QPainter painter(this);

    QPen pen(Qt::white);
    pen.setWidth(2);
    pen.setStyle(Qt::SolidLine);
    painter.setPen(pen);

    QRect rect = this->rect();

    int boarderRadius = 10;
    painter.drawRoundedRect(rect, boarderRadius, boarderRadius);
}

void ItemInfo::onEditButtonClicked(){
    bool isReadOnly = nameEdit->isReadOnly();

    nameEdit->setReadOnly(!isReadOnly);
    lengthEdit->setReadOnly(!isReadOnly);
    widthEdit->setReadOnly(!isReadOnly);
    heightEdit->setReadOnly(!isReadOnly);

    QPushButton* editButton = qobject_cast<QPushButton*>(sender());
    if(isReadOnly){
        editButton->setText("save");
        qDebug() << "editing";
    } else {    // 修改 edit
        editButton->setText("edit");

        QString id_to_update = id_edit->text();
        QString name_to_update = nameEdit->text();
        QString length_to_update = lengthEdit->text();
        QString width_to_update = widthEdit->text();
        QString height_to_update = heightEdit->text();


        qDebug() << "ID: " << id_to_update
                 << "Name: " << name_to_update
                 << "Length: " << length_to_update
                 << "Width: " << width_to_update
                 << "Height: " << height_to_update;

        QString table_name = "dataset";
        bool success = dbConn::instance().update_item_with_id( table_name
                                              , id_to_update
                                              , name_to_update
                                              , length_to_update
                                              , height_to_update
                                              , width_to_update
                                              ) ;

        if (success)
            QMessageBox::information(this, "Success", "Item updated successfully!");
        else
            QMessageBox::warning(this, "Error", "Failed to update item!");


        qDebug() << "saved";
    }

    qDebug() << getName();
}

void ItemInfo::set_id(const QString& id) { id_label->setText(id); }
void ItemInfo::setName(const QString& name) { nameLabel->setText(name); }
void ItemInfo::setLength(const QString& length) { lengthLabel->setText(length); }
void ItemInfo::setWidth(const QString& width) { widthLabel->setText(width); }
void ItemInfo::setHeight(const QString& height) { heightLabel->setText(height); }
void ItemInfo::setImage(const QPixmap& image) { imageLabel->setPixmap(image.scaled(150, 150, Qt::KeepAspectRatio)); }

QString ItemInfo::get_id() const { return " "; }
QString ItemInfo::getName() const { return nameEdit->text(); }
QString ItemInfo::getLength() const { return lengthEdit->text(); }
QString ItemInfo::getWidth() const { return widthEdit->text(); }
QString ItemInfo::getHeight() const { return heightEdit->text(); }
QPixmap ItemInfo::getImage() const { return imageLabel->pixmap(); }
