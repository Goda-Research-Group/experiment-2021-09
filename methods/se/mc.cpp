
namespace SE {

std::pair<int, Eigen::MatrixXd> MC::estimate(int N, std::mt19937& mt) {
    std::vector<Eigen::MatrixXd> xs(N);
    for (int ni = 0; ni < N; ni++) {
        xs[ni] = model.xRvs(mt);
    }

    Eigen::MatrixXd fs(model.fSize(), N);
    for (int ni = 0; ni < N; ni++) {
        fs.col(ni) = model.f(xs[ni]);
    }
    return {N, fs.rowwise().mean()};
}

Eigen::MatrixXd MC::getTheoretical() const {
    return model.getTheoretical();
}

}  // namespace SE
