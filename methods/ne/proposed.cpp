
namespace NE {

std::vector<int> Proposed::nSplit(int N, int K) {
    int res = 0;
    for (; pow(res, 2.0 * K) <= N; res++)
        ;
    return std::vector<int>(K, res - 1);
}

int Proposed::nAdjust(const std::vector<int>& ns) {
    int K = ns.size();
    int result = 1;
    for (int ki = 0; ki < K; ki++) {
        result *= ns[ki];
    }
    return result * result;
}

std::vector<std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>>
Proposed::grouping(
    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>& xys,
    const std::vector<int>& ns) {
    int N = xys.size();
    int K = xys[0].second.rows();

    int chunkSize = N;
    for (int ki = 0; ki < K; ki++) {
        for (int ci = 0; ci < N / chunkSize; ci++) {
            sort(xys.begin() + chunkSize * ci,
                 xys.begin() + chunkSize * ci + chunkSize,
                 [&ki](auto const& lhs, auto const& rhs) {
                     return lhs.second(ki, 0) < rhs.second(ki, 0);
                 });
        }
        chunkSize /= ns[ki];
    }

    std::vector<std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>>
        xyChunks(N / chunkSize,
                 std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>(
                     chunkSize));
    for (int ci = 0; ci < N / chunkSize; ci++) {
        for (int ni = 0; ni < chunkSize; ni++) {
            xyChunks[ci][ni] = xys[ci * chunkSize + ni];
        }
    }
    return xyChunks;
}

Eigen::MatrixXd Proposed::xTildes(
    const std::vector<std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>>&
        xyChunks) {
    int chunkNum = xyChunks.size();
    int chunkSize = xyChunks[0].size();
    int F = xyChunks[0][0].first.rows();

    Eigen::MatrixXd result(chunkNum * chunkSize, F);

    for (int ci = 0; ci < chunkNum; ci++) {
        for (int fi = 0; fi < F; fi++) {
            double average = 0;
            for (int ni = 0; ni < chunkSize; ni++) {
                average += xyChunks[ci][ni].first(fi, 0);
            }
            average /= chunkSize;
            for (int ni = 0; ni < chunkSize; ni++) {
                result(ci * chunkSize + ni, fi) = average;
            }
        }
    }
    return result.transpose();
}

Eigen::MatrixXd ProposedWithRegression::xTildes(
    const std::vector<std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>>&
        xyChunks) {
    int chunkNum = xyChunks.size();
    int chunkSize = xyChunks[0].size();
    int F = xyChunks[0][0].first.rows();
    int K = xyChunks[0][0].second.rows();

    Eigen::MatrixXd result(chunkNum * chunkSize, F);

    for (int ci = 0; ci < chunkNum; ci++) {
        for (int fi = 0; fi < F; fi++) {
            Eigen::MatrixXd M(chunkSize, K + 1);
            Eigen::VectorXd v(chunkSize);
            Eigen::VectorXd cHat(K + 1);
            Eigen::VectorXd xTildeChunk(chunkSize);
            for (int ni = 0; ni < chunkSize; ni++) {
                M(ni, 0) = 1;
                for (int ki = 0; ki < K; ki++) {
                    M(ni, ki + 1) = xyChunks[ci][ni].second(ki, 0);
                }
            }
            for (int ni = 0; ni < chunkSize; ni++) {
                v(ni) = xyChunks[ci][ni].first(fi, 0);
            }
            cHat = (M.transpose() * M).llt().solve(M.transpose() * v);
            xTildeChunk = M * cHat;
            result.block(ci * chunkSize, fi, chunkSize, 1) = xTildeChunk;
        }
    }
    return result.transpose();
}

Eigen::MatrixXd Proposed::smoothing(const Eigen::MatrixXd& xs,
                                    const Eigen::MatrixXd& ys) {
    std::vector<int> ns = nSplit(ys.cols(), ys.rows());
    int N = nAdjust(ns);

    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> xys(N);
    for (int ni = 0; ni < N; ni++) {
        xys[ni] = {xs.col(ni), ys.col(ni)};
    }

    return xTildes(grouping(xys, ns));
}

}  // namespace NE
