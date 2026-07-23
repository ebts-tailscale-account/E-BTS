#include "frame_view.h"

#include <QPainter>

namespace e_bts::gui {

FrameView::FrameView(QQuickItem *parent) : QQuickPaintedItem(parent) {}

void FrameView::setFrame(const QImage &frame) {
    frame_ = frame;
    update();
}

void FrameView::paint(QPainter *painter) {
    if (frame_.isNull()) {
        return;
    }

    const QSizeF bounds = boundingRect().size();
    const QSizeF fitted = frame_.size().scaled(bounds.toSize(), Qt::KeepAspectRatio);
    const QRectF target(QPointF((bounds.width() - fitted.width()) / 2.0, (bounds.height() - fitted.height()) / 2.0),
                        fitted);

    painter->setRenderHint(QPainter::SmoothPixmapTransform, false);
    painter->drawImage(target, frame_);
}

} // namespace e_bts::gui
