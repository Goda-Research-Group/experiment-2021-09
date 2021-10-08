
namespace NE {

TestCase::TestCase() {}

Eigen::MatrixXd TestCase::f(const Eigen::MatrixXd& gy) const {
    return gy.array().abs().log();
}

Eigen::MatrixXd TestCase::g(const std::vector<Eigen::MatrixXd>& yz) const {
    Eigen::MatrixXd result(1, 1);
    result << sqrt(2 / M_PI) * exp(-2 * (yz[0] - yz[1]).squaredNorm());
    return result;
}

int TestCase::fSize() const {
    return 1;
}

int TestCase::gSize() const {
    return 1;
}

Eigen::MatrixXd TestCase::yRvs(std::mt19937& mt) const {
    std::uniform_real_distribution<> y_g(-1.0, 1.0);
    Eigen::MatrixXd result(1, 1);
    result(0, 0) = y_g(mt);
    return result;
}

Eigen::MatrixXd TestCase::zRvs(const Eigen::MatrixXd&, std::mt19937& mt) const {
    std::normal_distribution<> z_g(0.0, 1.0);
    Eigen::MatrixXd result(1, 1);
    result(0, 0) = z_g(mt);
    return result;
}

int TestCase::ySize() const {
    return 1;
}

int TestCase::zSize() const {
    return 1;
}

Eigen::MatrixXd TestCase::getTheoretical() const {
    Eigen::MatrixXd result(1, 1);
    result(0, 0) = (1.0 / 2.0) * log(2.0 / (5.0 * M_PI)) - (2.0 / 15.0);
    return result;
}

}  // namespace NE
