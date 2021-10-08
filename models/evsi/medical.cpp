
namespace EVSI {

Medical::thetaType Medical::matrixToStruct(
    const Eigen::MatrixXd& thetaMatrix) const {
    thetaType thetaStruct;
    thetaStruct.L = thetaMatrix(0, 0);
    thetaStruct.Q_E = thetaMatrix(1, 0);
    thetaStruct.Q_SE = thetaMatrix(2, 0);
    thetaStruct.C_E = thetaMatrix(3, 0);
    thetaStruct.C_SE = thetaMatrix(4, 0);
    thetaStruct.C_T_1 = thetaMatrix(5, 0);
    thetaStruct.C_T_2 = thetaMatrix(6, 0);
    thetaStruct.C_T_3 = thetaMatrix(7, 0);
    thetaStruct.P_E_1 = thetaMatrix(8, 0);
    thetaStruct.logOR_E_2 = thetaMatrix(9, 0);
    thetaStruct.logOR_E_3 = thetaMatrix(10, 0);
    thetaStruct.P_E_2 = thetaMatrix(11, 0);
    thetaStruct.P_E_3 = thetaMatrix(12, 0);
    thetaStruct.P_SE_1 = thetaMatrix(13, 0);
    thetaStruct.P_SE_2 = thetaMatrix(14, 0);
    thetaStruct.P_SE_3 = thetaMatrix(15, 0);
    thetaStruct.lambda = thetaMatrix(16, 0);
    return thetaStruct;
}

Eigen::MatrixXd Medical::structToMatrix(const thetaType& thetaStruct) const {
    Eigen::MatrixXd thetaMatrix(17, 1);
    thetaMatrix(0, 0) = thetaStruct.L;
    thetaMatrix(1, 0) = thetaStruct.Q_E;
    thetaMatrix(2, 0) = thetaStruct.Q_SE;
    thetaMatrix(3, 0) = thetaStruct.C_E;
    thetaMatrix(4, 0) = thetaStruct.C_SE;
    thetaMatrix(5, 0) = thetaStruct.C_T_1;
    thetaMatrix(6, 0) = thetaStruct.C_T_2;
    thetaMatrix(7, 0) = thetaStruct.C_T_3;
    thetaMatrix(8, 0) = thetaStruct.P_E_1;
    thetaMatrix(9, 0) = thetaStruct.logOR_E_2;
    thetaMatrix(10, 0) = thetaStruct.logOR_E_3;
    thetaMatrix(11, 0) = thetaStruct.P_E_2;
    thetaMatrix(12, 0) = thetaStruct.P_E_3;
    thetaMatrix(13, 0) = thetaStruct.P_SE_1;
    thetaMatrix(14, 0) = thetaStruct.P_SE_2;
    thetaMatrix(15, 0) = thetaStruct.P_SE_3;
    thetaMatrix(16, 0) = thetaStruct.lambda;
    return thetaMatrix;
}

double Medical::sigmoid(double x) const {
    return 1.0 / (1.0 + exp(-x));
}

double Medical::logit(double p) const {
    return log(p / (1.0 - p));
}

Eigen::MatrixXd Medical::f_d(const Eigen::MatrixXd& thetaMatrix) const {
    Eigen::MatrixXd f(3, 1);
    thetaType theta = matrixToStruct(thetaMatrix);
    f(0, 0) =
        theta.P_SE_1 * theta.P_E_1 *
            (theta.lambda * (theta.L * (1.0 + theta.Q_E) / 2.0 - theta.Q_SE) -
             (theta.C_SE + theta.C_E)) +
        theta.P_SE_1 * (1.0 - theta.P_E_1) *
            (theta.lambda * (theta.L - theta.Q_SE) - theta.C_SE) +
        (1.0 - theta.P_SE_1) * theta.P_E_1 *
            (theta.lambda * theta.L * (1.0 + theta.Q_E) / 2.0 - theta.C_E) +
        (1.0 - theta.P_SE_1) * (1.0 - theta.P_E_1) * theta.lambda * theta.L -
        theta.C_T_1;
    f(1, 0) =
        theta.P_SE_2 * theta.P_E_2 *
            (theta.lambda * (theta.L * (1.0 + theta.Q_E) / 2.0 - theta.Q_SE) -
             (theta.C_SE + theta.C_E)) +
        theta.P_SE_2 * (1.0 - theta.P_E_2) *
            (theta.lambda * (theta.L - theta.Q_SE) - theta.C_SE) +
        (1.0 - theta.P_SE_2) * theta.P_E_2 *
            (theta.lambda * theta.L * (1.0 + theta.Q_E) / 2.0 - theta.C_E) +
        (1.0 - theta.P_SE_2) * (1.0 - theta.P_E_2) * theta.lambda * theta.L -
        theta.C_T_2;
    f(2, 0) =
        theta.P_SE_3 * theta.P_E_3 *
            (theta.lambda * (theta.L * (1.0 + theta.Q_E) / 2.0 - theta.Q_SE) -
             (theta.C_SE + theta.C_E)) +
        theta.P_SE_3 * (1.0 - theta.P_E_3) *
            (theta.lambda * (theta.L - theta.Q_SE) - theta.C_SE) +
        (1.0 - theta.P_SE_3) * theta.P_E_3 *
            (theta.lambda * theta.L * (1.0 + theta.Q_E) / 2.0 - theta.C_E) +
        (1.0 - theta.P_SE_3) * (1.0 - theta.P_E_3) * theta.lambda * theta.L -
        theta.C_T_3;
    return f;
}

int Medical::dSize() const {
    return 3;
}

Eigen::MatrixXd Medical::thetaRvs(std::mt19937& mt) const {
    thetaType theta;

    std::normal_distribution<> l_dist(30, sqrt(25.0));
    theta.L = l_dist(mt);
    std::normal_distribution<> q_e_dist(0.6, sqrt(1.0 / 36.0));
    theta.Q_E = sigmoid(q_e_dist(mt));
    std::normal_distribution<> q_se_dist(0.7, sqrt(0.01));
    theta.Q_SE = q_se_dist(mt);
    std::normal_distribution<> c_e_dist(200000, sqrt(100000000));
    theta.C_E = c_e_dist(mt);
    std::normal_distribution<> c_se_dist(100000, sqrt(100000000));
    theta.C_SE = c_se_dist(mt);

    theta.C_T_1 = 0;
    std::normal_distribution<> c_t_2_dist(1.5 * 10000, sqrt(300.0));
    theta.C_T_2 = c_t_2_dist(mt);
    std::normal_distribution<> c_t_3_dist(
        2.0 * 10000 + (100.0 / 300.0) * (theta.C_T_2 - 1.5 * 10000),
        sqrt(150.0 - 100.0 * 100.0 / 300.0));
    theta.C_T_3 = c_t_3_dist(mt);

    std::gamma_distribution<> p_e_1_dist_a(15.0, 1.0);
    std::gamma_distribution<> p_e_1_dist_b(85.0, 1.0);
    double ya = p_e_1_dist_a(mt);
    double yb = p_e_1_dist_b(mt);
    theta.P_E_1 = ya / (ya + yb);
    std::normal_distribution<> log_or_e_2_dist(-1.5, sqrt(0.11));
    theta.logOR_E_2 = log_or_e_2_dist(mt);
    std::normal_distribution<> log_or_e_3_dist(
        -1.75 + (0.02 / 0.11) * (theta.logOR_E_2 + 1.5),
        sqrt(0.06 - 0.02 * 0.02 / 0.11));
    theta.logOR_E_3 = log_or_e_3_dist(mt);
    theta.P_E_2 = sigmoid(theta.logOR_E_2 + logit(theta.P_E_1));
    theta.P_E_3 = sigmoid(theta.logOR_E_3 + logit(theta.P_E_1));

    theta.P_SE_1 = 0;
    std::normal_distribution<> logit_p_se_2_dist(-1.4, sqrt(0.10));
    theta.P_SE_2 = sigmoid(logit_p_se_2_dist(mt));
    std::normal_distribution<> logit_p_se_3_dist(
        -1.1 + (0.05 / 0.10) * (logit(theta.P_SE_2) + 1.4),
        sqrt(0.25 - 0.05 * 0.05 / 0.10));
    theta.P_SE_3 = sigmoid(logit_p_se_3_dist(mt));

    theta.lambda = 75000;

    return structToMatrix(theta);
}

Eigen::MatrixXd Medical::thetaRvsGivenPhi(const Eigen::MatrixXd&,
                                          std::mt19937&) const {
    assert(false);
}

Eigen::MatrixXd Medical::phiRvs(const Eigen::MatrixXd& thetaMatrix,
                                std::mt19937& mt) const {
    thetaType theta = matrixToStruct(thetaMatrix);
    int N_p = 100;
    std::normal_distribution<> phi0_dist(theta.logOR_E_3, sqrt(4.0 / N_p));
    std::normal_distribution<> phi1_dist(theta.C_T_3, sqrt(1.0 * 10000 / N_p));
    std::binomial_distribution<> phi2_dist(N_p, theta.P_SE_3);
    Eigen::MatrixXd phi(3, 1);
    phi << phi0_dist(mt), phi1_dist(mt), phi2_dist(mt);
    return phi;
}

int Medical::phiSize() const {
    return 3;
}

Eigen::MatrixXd Medical::getTheoretical() const {
    Eigen::MatrixXd result(1, 1);
    result(0, 0) = 1031;
    return result;
}

}  // namespace EVSI
