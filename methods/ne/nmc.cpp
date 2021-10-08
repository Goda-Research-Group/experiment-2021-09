
namespace NE {

Eigen::MatrixXd NMC::gamma(const Eigen::MatrixXd& y, int M, std::mt19937& mt) {
    Eigen::MatrixXd gs(model.gSize(), M);
    for (int ni = 0; ni < M; ni++) {
        auto z = model.zRvs(y, mt);
        gs.col(ni) = model.g({y, z});
    }
    return gs.rowwise().mean();
}

int NMC::adjustM(int N) {
    int M = 0;
    for (; M * M <= N; M++)
        ;
    return M - 1;
}

std::pair<int, Eigen::MatrixXd> NMC::estimate(int N, std::mt19937& mt) {
    int M = adjustM(N);
    int B = N / M;

    Eigen::MatrixXd fs(model.fSize(), B);
    for (int ni = 0; ni < B; ni++) {
        auto y = model.yRvs(mt);
        fs.col(ni) = model.f(gamma(y, M, mt));
    }
    return {M * B, fs.rowwise().mean()};
}

Eigen::MatrixXd NMC::getTheoretical() const {
    return model.getTheoretical();
}

}  // namespace NE
