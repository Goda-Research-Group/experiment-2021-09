
#include <fstream>
#include <iostream>

namespace NE {

Eigen::MatrixXd Strong::smoothing(const Eigen::MatrixXd& xs,
                                  const Eigen::MatrixXd& ys) {
    int N = xs.cols();
    int F = xs.rows();
    int K = ys.rows();

    assert(K <= 3);

    std::string inputF = tmpPath + "/inputF.txt";
    std::ofstream ofsF(inputF);
    if (!ofsF) {
        assert(false);
    }
    ofsF << xs.transpose() << std::endl;

    std::string inputY = tmpPath + "/inputY.txt";
    std::ofstream ofsY(inputY);
    if (!ofsY) {
        assert(false);
    }
    ofsY << ys.transpose() << std::endl;

    int ret = system(("Rscript methods/ne/strong.R " + tmpPath).c_str());

    Eigen::MatrixXd xTilde;
    if (ret != 0) {
        xTilde = Eigen::MatrixXd(F, N);
        for (int fi = 0; fi < F; fi++) {
            for (int ni = 0; ni < N; ni++) {
                xTilde(fi, ni) = std::nan("");
            }
        }
    } else {
        xTilde = Eigen::MatrixXd(F, N);
        std::string outputF = tmpPath + "/outputF.txt";
        std::ifstream ifs(outputF);
        if (!ifs) {
            assert(false);
        }
        for (int ni = 0; ni < N; ni++) {
            for (int fi = 0; fi < F; fi++) {
                ifs >> xTilde(fi, ni);
            }
        }
    }
    return xTilde;
}

}  // namespace NE
