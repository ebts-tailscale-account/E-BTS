#ifndef E_BTS_GUI_FRAME_VIEW_H
#define E_BTS_GUI_FRAME_VIEW_H

#include <QImage>
#include <QQuickPaintedItem>

namespace e_bts::gui {

// QML item that letterboxes the most recently set QImage into its bounding
// rect, preserving aspect ratio -- the QML-side replacement for
// display_utils.h's LetterboxedFramePresenter (which composited onto a
// cv::Mat for an OpenCV/GLFW window). Register as EBts.FrameView.
class FrameView : public QQuickPaintedItem {
    Q_OBJECT

public:
    explicit FrameView(QQuickItem *parent = nullptr);

    void paint(QPainter *painter) override;

public slots:
    void setFrame(const QImage &frame);

private:
    QImage frame_;
};

} // namespace e_bts::gui

#endif // E_BTS_GUI_FRAME_VIEW_H
