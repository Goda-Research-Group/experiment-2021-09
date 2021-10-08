
namespace EVSI {

Eigen::MatrixXd NMC::modelFirstTerm::f(const Eigen::MatrixXd& gy) const {
    return gy.colwise().maxCoeff();
}

Eigen::MatrixXd NMC::modelFirstTerm::g(
    const std::vector<Eigen::MatrixXd>& yz) const {
    return modelEVSI.f_d(yz[1]);
}

int NMC::modelFirstTerm::fSize() const {
    return 1;
}

int NMC::modelFirstTerm::gSize() const {
    return modelEVSI.dSize();
}

Eigen::MatrixXd NMC::modelFirstTerm::yRvs(std::mt19937& mt) const {
    int c = sharedSamples.first++;
    sharedSamples.second[c] = modelEVSI.thetaRvs(mt);
    return modelEVSI.phiRvs(sharedSamples.second[c], mt);
}

Eigen::MatrixXd NMC::modelFirstTerm::zRvs(const Eigen::MatrixXd& y,
                                          std::mt19937& mt) const {
    return modelEVSI.thetaRvsGivenPhi(y, mt);
}

int NMC::modelFirstTerm::ySize() const {
    return modelEVSI.phiSize();
}

int NMC::modelFirstTerm::zSize() const {
    assert(false);
}

Eigen::MatrixXd NMC::modelFirstTerm::getTheoretical() const {
    assert(false);
}

Eigen::MatrixXd NMC::modelSecondTerm::f(const Eigen::MatrixXd& x) const {
    return modelEVSI.f_d(x);
}

int NMC::modelSecondTerm::fSize() const {
    return modelEVSI.dSize();
}

Eigen::MatrixXd NMC::modelSecondTerm::xRvs(std::mt19937&) const {
    int c = sharedSamples.first++;
    return sharedSamples.second[c];
}

Eigen::MatrixXd NMC::modelSecondTerm::getTheoretical() const {
    assert(false);
}

void NMC::initSharedSamples(int N, std::mt19937&) {
    sharedSamples.second = std::vector<Eigen::MatrixXd>(N);
}

std::pair<int, Eigen::MatrixXd> NMC::estimate(int N, std::mt19937& mt) {
    initSharedSamples(N, mt);
    sharedSamples.first = 0;
    auto ft = estimatorFirstTerm.estimate(N, mt);
    int nOuterSamples = sharedSamples.first;
    sharedSamples.first = 0;
    auto st = estimatorSecondTerm.estimate(nOuterSamples, mt);
    return {ft.first, ft.second - st.second.colwise().maxCoeff()};
}

Eigen::MatrixXd NMC::getTheoretical() const {
    return model.getTheoretical();
}

}  // namespace EVSI
