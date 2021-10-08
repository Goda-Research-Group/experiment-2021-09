
void visualize(std::vector<int> Ns,
               int R,
               Estimator& e,
               std::string outputFile,
               std::mt19937& mt) {
    FILE* fp1 = fopen(outputFile.c_str(), "w");
    if (fp1 == NULL) {
        throw std::runtime_error("Error: Cannot open the file.\n");
    }

    for (int n : Ns) {
        Eigen::MatrixXd errors(e.getTheoretical().rows(), R);
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> num(1, R);
        for (int ri = 0; ri < R; ri++) {
            auto result = e.estimate(n, mt);
            num(0, ri) = result.first;
            errors.col(ri) = result.second - e.getTheoretical();
        }
        double mse = (errors.colwise().squaredNorm()).mean();
        double smean = (errors.rowwise().mean()).squaredNorm();
        double var = mse - smean;
        int nmean = num.mean();
        fprintf(fp1, "%10d %+.12f %+.12f %+.12f\n", nmean, mse, smean, var);
    }

    fclose(fp1);
}
