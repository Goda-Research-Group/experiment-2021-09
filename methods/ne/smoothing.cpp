
#include <fstream>
#include <iostream>

namespace NE {

std::pair<int, Eigen::MatrixXd> Smoothing::estimate(int N, std::mt19937& mt) {
    std::vector<std::vector<Eigen::MatrixXd>> yz(N);
    for (int ni = 0; ni < N; ni++) {
        yz[ni] = model.psaRvs(mt);
    }

    Eigen::MatrixXd x(model.gSize(), N), y(model.ySize(), N);
    for (int ni = 0; ni < N; ni++) {
        x.col(ni) = model.g(yz[ni]);
        y.col(ni) = yz[ni][0];
    }

    Eigen::MatrixXd xTilde = smoothing(x, y);

    int nAct = xTilde.cols();
    Eigen::MatrixXd fXTilde(model.fSize(), nAct);
    for (int ni = 0; ni < nAct; ni++) {
        fXTilde.col(ni) = model.f(xTilde.col(ni));
    }

    return {nAct, fXTilde.rowwise().mean()};
}

Eigen::MatrixXd Smoothing::getTheoretical() const {
    return model.getTheoretical();
}

}  // namespace NE
