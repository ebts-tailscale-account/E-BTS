#ifndef E_BTS_GUI_FT_GRAPH_VIEW_H
#define E_BTS_GUI_FT_GRAPH_VIEW_H

#include <QMutex>
#include <QQuickPaintedItem>
#include <QString>
#include <QStringList>
#include <QVector>

namespace e_bts::gui {

// Rolling real-time line plot of three channels -- one instance drives the
// forces (Fx/Fy/Fz), another the torques (Mx/My/Mz). Same QQuickPaintedItem
// base as FrameView; addSample() is called from QML at the worker's decimated
// ~60 Hz display rate. A mutex guards the sample buffers because, unlike
// FrameView's single implicitly-shared QImage, QVector reallocation could race
// paint() (which may run on the scene-graph render thread). Register as
// EBts.FtGraphView.
class FtGraphView : public QQuickPaintedItem {
    Q_OBJECT
    Q_PROPERTY(QString title READ title WRITE setTitle NOTIFY titleChanged)
    Q_PROPERTY(QStringList seriesNames READ seriesNames WRITE setSeriesNames NOTIFY seriesNamesChanged)

public:
    explicit FtGraphView(QQuickItem *parent = nullptr);

    void paint(QPainter *painter) override;

    QString title() const { return title_; }
    void setTitle(const QString &title);

    QStringList seriesNames() const { return series_names_; }
    void setSeriesNames(const QStringList &names);

public slots:
    void addSample(double a, double b, double c);
    void clear();

signals:
    void titleChanged();
    void seriesNamesChanged();

private:
    QString title_;
    QStringList series_names_{"A", "B", "C"};

    QVector<double> s0_;
    QVector<double> s1_;
    QVector<double> s2_;
    mutable QMutex mutex_;

    static constexpr int kMaxPoints = 900; // ~15 s at 60 Hz
};

} // namespace e_bts::gui

#endif // E_BTS_GUI_FT_GRAPH_VIEW_H
