
namespace EVSI {

Eigen::MatrixXd TestCase::f_d(const Eigen::MatrixXd& theta) const {
    Eigen::MatrixXd result(2, 1);
    result(0, 0) = theta(0, 0);
    result(1, 0) = -theta(0, 0);
    return result;
}

int TestCase::dSize() const {
    return 2;
}

Eigen::MatrixXd TestCase::thetaRvs(std::mt19937& mt) const {
    static std::normal_distribution<> z_dist(0, 1);
    Eigen::MatrixXd result(1, 1);
    result << z_dist(mt);
    return result;
}

Eigen::MatrixXd TestCase::phiRvs(const Eigen::MatrixXd& theta,
                                 std::mt19937& mt) const {
    static std::normal_distribution<> z_dist(0, 1);
    Eigen::MatrixXd phi(phiSize(), 1);
    double cur = theta(0, 0);
    for (int ki = 0; ki < phiSize(); ki++) {
        phi(ki, 0) =
            cur / (ki + 2.0) + z_dist(mt) * sqrt((ki + 3.0) / (2 * ki + 4.0));
        cur += phi(ki, 0);
    }
    return phi;
}

int TestCase::phiSize() const {
    return M;
}

Eigen::MatrixXd TestCase::thetaRvsGivenPhi(const Eigen::MatrixXd& phi,
                                           std::mt19937& mt) const {
    static std::normal_distribution<> z_dist(0, 1);
    double cur = 0;
    for (int ki = 0; ki < phiSize(); ki++) {
        cur += phi(ki, 0);
    }
    Eigen::MatrixXd result(1, 1);
    result(0, 0) =
        cur / ((phiSize() - 1) + 2.0) +
        z_dist(mt) * sqrt((phiSize() - 1 + 3.0) / (2 * (phiSize() - 1) + 4.0));
    return result;
}

Eigen::MatrixXd TestCase::getTheoretical() const {
    Eigen::MatrixXd A(1, 1);
    A(0, 0) = sqrt(phiSize() / ((phiSize() + 1.0) * M_PI));
    return A;
}

}  // namespace EVSI
