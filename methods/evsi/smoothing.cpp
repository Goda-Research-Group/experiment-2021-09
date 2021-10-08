
namespace EVSI {

Eigen::MatrixXd Smoothing::modelFirstTerm::f(const Eigen::MatrixXd& gy) const {
    return gy.colwise().maxCoeff();
}

Eigen::MatrixXd Smoothing::modelFirstTerm::g(
    const std::vector<Eigen::MatrixXd>& yz) const {
    return modelEVSI.f_d(yz[1]);
}

int Smoothing::modelFirstTerm::fSize() const {
    return 1;
}

int Smoothing::modelFirstTerm::gSize() const {
    return modelEVSI.dSize();
}

Eigen::MatrixXd Smoothing::modelFirstTerm::yRvs(std::mt19937& mt) const {
    return modelEVSI.phiRvs(modelEVSI.thetaRvs(mt), mt);
}

Eigen::MatrixXd Smoothing::modelFirstTerm::zRvs(const Eigen::MatrixXd& y,
                                                std::mt19937& mt) const {
    return modelEVSI.thetaRvsGivenPhi(y, mt);
}

std::vector<Eigen::MatrixXd> Smoothing::modelFirstTerm::psaRvs(
    std::mt19937&) const {
    int c = sharedSamples.first++;
    return {sharedSamples.second[c][1], sharedSamples.second[c][0]};
}

int Smoothing::modelFirstTerm::ySize() const {
    return modelEVSI.phiSize();
}

int Smoothing::modelFirstTerm::zSize() const {
    assert(false);
}

Eigen::MatrixXd Smoothing::modelFirstTerm::getTheoretical() const {
    assert(false);
}

Eigen::MatrixXd Smoothing::modelSecondTerm::f(const Eigen::MatrixXd& x) const {
    return modelEVSI.f_d(x);
}

int Smoothing::modelSecondTerm::fSize() const {
    return modelEVSI.dSize();
}

Eigen::MatrixXd Smoothing::modelSecondTerm::xRvs(std::mt19937&) const {
    int c = sharedSamples.first++;
    return sharedSamples.second[c][0];
}

Eigen::MatrixXd Smoothing::modelSecondTerm::getTheoretical() const {
    assert(false);
}

void Smoothing::initSharedSamples(int N, std::mt19937& mt) {
    sharedSamples.second = std::vector<std::vector<Eigen::MatrixXd>>(N);
    for (int ni = 0; ni < N; ni++) {
        sharedSamples.second[ni] = std::vector<Eigen::MatrixXd>(2);
        sharedSamples.second[ni][0] = model.thetaRvs(mt);
        sharedSamples.second[ni][1] =
            model.phiRvs(sharedSamples.second[ni][0], mt);
    }
}

std::pair<int, Eigen::MatrixXd> Smoothing::estimate(int N, std::mt19937& mt) {
    initSharedSamples(N, mt);
    sharedSamples.first = 0;
    auto ft = estimatorFirstTerm.estimate(N, mt);
    sharedSamples.first = 0;
    auto st = estimatorSecondTerm.estimate(ft.first, mt);
    return {ft.first, ft.second - st.second.colwise().maxCoeff()};
}

Eigen::MatrixXd Smoothing::getTheoretical() const {
    return model.getTheoretical();
}

}  // namespace EVSI
