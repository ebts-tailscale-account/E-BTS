#include "ft_graph_view.h"

#include <algorithm>

#include <QColor>
#include <QFont>
#include <QPainter>
#include <QPainterPath>
#include <QPen>

namespace e_bts::gui {

namespace {

// Fx/Mx = red, Fy/My = green (matches the app accent), Fz/Mz = blue.
const QColor kSeriesColors[3] = {QColor("#E53935"), QColor("#43A047"), QColor("#42A5F5")};

} // namespace

FtGraphView::FtGraphView(QQuickItem *parent) : QQuickPaintedItem(parent) {}

void FtGraphView::setTitle(const QString &title) {
    if (title != title_) {
        title_ = title;
        emit titleChanged();
        update();
    }
}

void FtGraphView::setSeriesNames(const QStringList &names) {
    if (names != series_names_) {
        series_names_ = names;
        emit seriesNamesChanged();
        update();
    }
}

void FtGraphView::addSample(double a, double b, double c) {
    {
        QMutexLocker lock(&mutex_);
        s0_.append(a);
        s1_.append(b);
        s2_.append(c);
        while (s0_.size() > kMaxPoints) {
            s0_.removeFirst();
            s1_.removeFirst();
            s2_.removeFirst();
        }
    }
    update();
}

void FtGraphView::clear() {
    {
        QMutexLocker lock(&mutex_);
        s0_.clear();
        s1_.clear();
        s2_.clear();
    }
    update();
}

void FtGraphView::paint(QPainter *painter) {
    QVector<double> a;
    QVector<double> b;
    QVector<double> c;
    {
        QMutexLocker lock(&mutex_);
        a = s0_;
        b = s1_;
        c = s2_;
    }
    const QVector<double> *series[3] = {&a, &b, &c};

    const qreal w = width();
    const qreal h = height();
    painter->fillRect(QRectF(0, 0, w, h), QColor("#161616"));

    const qreal margin_l = 8, margin_r = 8, margin_t = 6, margin_b = 6;
    const qreal gx = margin_l, gy = margin_t;
    const qreal gw = w - margin_l - margin_r;
    const qreal gh = h - margin_t - margin_b;
    if (gw <= 1.0 || gh <= 1.0) {
        return;
    }

    // Autoscale to the data, always including zero, with 10% headroom.
    double mn = 0.0, mx = 0.0;
    bool any = false;
    for (const QVector<double> *s : series) {
        for (double v : *s) {
            if (!any) {
                mn = mx = v;
                any = true;
            } else {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
        }
    }
    if (!any) {
        mn = -1.0;
        mx = 1.0;
    }
    mn = std::min(mn, 0.0);
    mx = std::max(mx, 0.0);
    if (mx - mn < 1e-6) {
        mx += 1.0;
        mn -= 1.0;
    }
    const double pad = (mx - mn) * 0.1;
    mn -= pad;
    mx += pad;

    const auto ymap = [&](double v) { return gy + gh * (1.0 - (v - mn) / (mx - mn)); };
    const auto xmap = [&](int i, int n) { return (n <= 1) ? gx : gx + gw * double(i) / double(n - 1); };

    // Zero line.
    painter->setPen(QPen(QColor("#3A3A3A"), 1, Qt::DashLine));
    painter->drawLine(QPointF(gx, ymap(0.0)), QPointF(gx + gw, ymap(0.0)));

    // Traces.
    for (int k = 0; k < 3; ++k) {
        const QVector<double> &s = *series[k];
        if (s.size() < 2) {
            continue;
        }
        QPen pen(kSeriesColors[k]);
        pen.setWidthF(1.4);
        painter->setPen(pen);

        QPainterPath path;
        const int n = s.size();
        path.moveTo(xmap(0, n), ymap(s[0]));
        for (int i = 1; i < n; ++i) {
            path.lineTo(xmap(i, n), ymap(s[i]));
        }
        painter->drawPath(path);
    }

    // Title (top-left).
    QFont title_font = painter->font();
    title_font.setPixelSize(11);
    painter->setFont(title_font);
    painter->setPen(QColor("#EAEAEA"));
    painter->drawText(QRectF(gx + 2, gy, gw, 14), Qt::AlignLeft | Qt::AlignVCenter, title_);

    // Legend + current value per channel.
    QFont mono = painter->font();
    mono.setFamily("monospace");
    painter->setFont(mono);
    qreal ly = gy + 16;
    for (int k = 0; k < 3; ++k) {
        painter->setPen(kSeriesColors[k]);
        const QString name = (k < series_names_.size()) ? series_names_[k] : QString("s%1").arg(k);
        const QString value = series[k]->isEmpty() ? QStringLiteral("--") : QString::number(series[k]->last(), 'f', 2);
        painter->drawText(QRectF(gx + 2, ly, 160, 14), Qt::AlignLeft, QString("%1 %2").arg(name).arg(value));
        ly += 14;
    }

    // Y-range labels (right edge).
    painter->setPen(QColor("#9E9E9E"));
    painter->drawText(QRectF(gx, gy, gw, 12), Qt::AlignRight | Qt::AlignTop, QString::number(mx, 'f', 1));
    painter->drawText(QRectF(gx, gy + gh - 12, gw, 12), Qt::AlignRight | Qt::AlignBottom, QString::number(mn, 'f', 1));
}

} // namespace e_bts::gui
