// E_BTS_pixel_to_mm_probe -- read pixels on stdin, print millimetres on stdout.
//
// This exists so that ml/check_pixel_to_mm_port.py can prove, on real numbers,
// that src/pixel_to_mm.h and ml/undistort.py evaluate the SAME calibration the
// same way. src/pixel_to_mm.h is a second implementation of a model that already
// had one, which is precisely the situation ml/undistort.py's header warns about:
// "a calibration that is evaluated by a slightly different polynomial than the one
// that was fitted is wrong in a way nothing will ever print". This prints it.
//
//     echo "320 240" | ./E_BTS_pixel_to_mm_probe
//     python3 ml/check_pixel_to_mm_port.py            # the actual check
//
// Output is ROBOT BASE millimetres, because that is what ml/undistort.py's
// pixel_to_mm returns and the point here is to compare like with like. Pixels
// outside the calibrated field print "nan nan", which is itself part of what must
// agree.
//
// Pass an elastomer-origin file as the second argument to get PAD-CENTRED
// millimetres instead -- exactly the call the GUI makes (pixel_to_pad_mm), so a
// freshly taught origin can be checked end to end without opening the camera:
//
//     echo "320 240" | ./E_BTS_pixel_to_mm_probe \
//         calibration/pixel_to_mm.json calibration/elastomer_origin.json

#include <iomanip>
#include <iostream>
#include <string>

#include <QCoreApplication>
#include <QString>

#include "pixel_to_mm.h"

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);
    const QString calibration_path =
        argc > 1 ? QString::fromLocal8Bit(argv[1]) : e_bts::default_pixel_to_mm_path();

    // "/dev/null" as the default origin: no origin file means robot-frame output,
    // which is what check_pixel_to_mm_port.py compares against.
    const QString origin_path = argc > 2 ? QString::fromLocal8Bit(argv[2]) : QStringLiteral("/dev/null");

    const e_bts::PixelToMm calibration = e_bts::load_pixel_to_mm(calibration_path, origin_path);
    if (!calibration.loaded) {
        std::cerr << "cannot load calibration: " << calibration.error.toStdString() << '\n';
        return 1;
    }
    if (argc > 2) {
        if (!calibration.has_origin) {
            std::cerr << "cannot load origin: " << calibration.origin_error.toStdString() << '\n';
            return 1;
        }
        std::cerr << "origin loaded: pad centre at robot (" << calibration.origin_mm.x << ", "
                  << calibration.origin_mm.y << ") mm; output is PAD-CENTRED\n";
    }

    double u = 0.0;
    double v = 0.0;
    std::cout << std::fixed << std::setprecision(9);
    while (std::cin >> u >> v) {
        cv::Point2d mm;
        const bool ok = argc > 2 ? calibration.pixel_to_pad_mm(cv::Point2d(u, v), mm)
                                 : calibration.pixel_to_mm(cv::Point2d(u, v), mm);
        if (ok) {
            std::cout << mm.x << ' ' << mm.y << '\n';
        } else {
            std::cout << "nan nan\n";
        }
    }
    return 0;
}
