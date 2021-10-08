#ifndef _ALL_CPP
#define _ALL_CPP

#include <inttypes.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
#include <random>
#include <vector>

class Estimator {
   public:
    virtual std::pair<int, Eigen::MatrixXd> estimate(int N,
                                                     std::mt19937& mt) = 0;
    virtual Eigen::MatrixXd getTheoretical() const = 0;
};

namespace SE {

class Model {
   public:
    Model() {}
    virtual Eigen::MatrixXd f(const Eigen::MatrixXd& x) const = 0;
    virtual int fSize() const = 0;
    virtual Eigen::MatrixXd xRvs(std::mt19937& mt) const = 0;
    virtual Eigen::MatrixXd getTheoretical() const = 0;
};

class MC : public ::Estimator {
   private:
    Model& model;

   public:
    MC(Model& model) : model{model} {}
    std::pair<int, Eigen::MatrixXd> estimate(int N, std::mt19937& mt);
    Eigen::MatrixXd getTheoretical() const;
};

}  // namespace SE

namespace NE {

class Model {
   public:
    Model() {}
    virtual Eigen::MatrixXd f(const Eigen::MatrixXd& gy) const = 0;
    virtual Eigen::MatrixXd g(const std::vector<Eigen::MatrixXd>& yz) const = 0;
    virtual int fSize() const = 0;
    virtual int gSize() const = 0;
    virtual Eigen::MatrixXd yRvs(std::mt19937& mt) const = 0;
    virtual Eigen::MatrixXd zRvs(const Eigen::MatrixXd& y,
                                 std::mt19937& mt) const = 0;
    virtual int ySize() const = 0;
    virtual int zSize() const = 0;
    virtual std::vector<Eigen::MatrixXd> psaRvs(std::mt19937& mt) const {
        auto y = yRvs(mt);
        auto z = zRvs(y, mt);
        return {y, z};
    }
    virtual Eigen::MatrixXd getTheoretical() const = 0;
};

class TestCase : public Model {
   public:
    TestCase();
    Eigen::MatrixXd f(const Eigen::MatrixXd& gy) const;
    Eigen::MatrixXd g(const std::vector<Eigen::MatrixXd>& yz) const;
    int fSize() const;
    int gSize() const;
    Eigen::MatrixXd yRvs(std::mt19937& mt) const;
    Eigen::MatrixXd zRvs(const Eigen::MatrixXd& y, std::mt19937& mt) const;
    int ySize() const;
    int zSize() const;
    Eigen::MatrixXd getTheoretical() const;
};

class NMC : public ::Estimator {
   private:
    Model& model;
    int adjustM(int N);
    Eigen::MatrixXd gamma(const Eigen::MatrixXd& y, int M, std::mt19937& mt);

   public:
    NMC(Model& model) : model{model} {}
    std::pair<int, Eigen::MatrixXd> estimate(int N, std::mt19937& mt);
    Eigen::MatrixXd getTheoretical() const;
};

class Smoothing : public ::Estimator {
   private:
    Model& model;
    virtual Eigen::MatrixXd smoothing(const Eigen::MatrixXd& xs,
                                      const Eigen::MatrixXd& ys) = 0;

   public:
    Smoothing(Model& model) : model{model} {}
    std::pair<int, Eigen::MatrixXd> estimate(int N, std::mt19937& mt);
    Eigen::MatrixXd getTheoretical() const;
};

class Proposed : public Smoothing {
   private:
    Eigen::MatrixXd smoothing(const Eigen::MatrixXd& xs,
                              const Eigen::MatrixXd& ys);
    std::vector<int> nSplit(int N, int K);
    int nAdjust(const std::vector<int>& no);
    std::vector<std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>>
    grouping(std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>& xys,
             const std::vector<int>& no);
    virtual Eigen::MatrixXd xTildes(
        const std::vector<
            std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>>&
            xyChunks);

   public:
    Proposed(Model& model) : Smoothing{model} {}
};

class ProposedWithRegression : public Proposed {
   private:
    Eigen::MatrixXd xTildes(
        const std::vector<
            std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>>&
            xyChunks);

   public:
    ProposedWithRegression(Model& model) : Proposed{model} {}
};

class Strong : public Smoothing {
   private:
    Eigen::MatrixXd smoothing(const Eigen::MatrixXd& xs,
                              const Eigen::MatrixXd& ys);
    const std::string tmpPath;

   public:
    Strong(Model& model, std::string tmpPath)
        : Smoothing{model}, tmpPath{tmpPath} {}
};

}  // namespace NE

namespace EVSI {

class Model {
   public:
    Model() {}
    virtual Eigen::MatrixXd f_d(const Eigen::MatrixXd& theta) const = 0;
    virtual int dSize() const = 0;
    virtual Eigen::MatrixXd thetaRvs(std::mt19937& mt) const = 0;
    virtual Eigen::MatrixXd phiRvs(const Eigen::MatrixXd& theta,
                                   std::mt19937& mt) const = 0;
    virtual int phiSize() const = 0;
    virtual Eigen::MatrixXd thetaRvsGivenPhi(const Eigen::MatrixXd& phi,
                                             std::mt19937& mt) const = 0;
    virtual Eigen::MatrixXd getTheoretical() const = 0;
};

class TestCase : public Model {
   private:
    const int M;

   public:
    TestCase(int M) : M{M} {}
    Eigen::MatrixXd f_d(const Eigen::MatrixXd& theta) const;
    int dSize() const;
    Eigen::MatrixXd thetaRvs(std::mt19937& mt) const;
    Eigen::MatrixXd phiRvs(const Eigen::MatrixXd& theta,
                           std::mt19937& mt) const;
    int phiSize() const;
    Eigen::MatrixXd thetaRvsGivenPhi(const Eigen::MatrixXd& phi,
                                     std::mt19937& mt) const;
    Eigen::MatrixXd getTheoretical() const;
};

class Medical : public Model {
   private:
    typedef struct {
        double L, Q_E, Q_SE, C_E, C_SE, C_T_1, C_T_2, C_T_3, P_E_1, logOR_E_2,
            logOR_E_3, P_E_2, P_E_3, P_SE_1, P_SE_2, P_SE_3, lambda;
    } thetaType;
    thetaType matrixToStruct(const Eigen::MatrixXd& thetaMatrix) const;
    Eigen::MatrixXd structToMatrix(const thetaType& thetaStruct) const;
    double sigmoid(double x) const;
    double logit(double p) const;

   public:
    Medical() {}
    Eigen::MatrixXd f_d(const Eigen::MatrixXd& theta) const;
    int dSize() const;
    Eigen::MatrixXd thetaRvs(std::mt19937& mt) const;
    Eigen::MatrixXd phiRvs(const Eigen::MatrixXd& theta,
                           std::mt19937& mt) const;
    int phiSize() const;
    Eigen::MatrixXd thetaRvsGivenPhi(const Eigen::MatrixXd& phi,
                                     std::mt19937& mt) const;
    Eigen::MatrixXd getTheoretical() const;
};

class NMC : public ::Estimator {
   private:
    Model& model;
    class modelFirstTerm : public NE::Model {
       private:
        EVSI::Model& modelEVSI;
        std::pair<int, std::vector<Eigen::MatrixXd>>& sharedSamples;

       public:
        modelFirstTerm(
            EVSI::Model& model,
            std::pair<int, std::vector<Eigen::MatrixXd>>& sharedSamples)
            : Model{}, modelEVSI{model}, sharedSamples{sharedSamples} {};
        Eigen::MatrixXd f(const Eigen::MatrixXd& gy) const;
        Eigen::MatrixXd g(const std::vector<Eigen::MatrixXd>& yz) const;
        int fSize() const;
        int gSize() const;
        Eigen::MatrixXd yRvs(std::mt19937& mt) const;
        Eigen::MatrixXd zRvs(const Eigen::MatrixXd& y, std::mt19937& mt) const;
        int ySize() const;
        int zSize() const;
        Eigen::MatrixXd getTheoretical() const;
    } modelFirstTerm;
    NE::NMC estimatorFirstTerm;

    class modelSecondTerm : public SE::Model {
       private:
        EVSI::Model& modelEVSI;
        std::pair<int, std::vector<Eigen::MatrixXd>>& sharedSamples;

       public:
        modelSecondTerm(
            EVSI::Model& model,
            std::pair<int, std::vector<Eigen::MatrixXd>>& sharedSamples)
            : modelEVSI{model}, sharedSamples{sharedSamples} {};
        Eigen::MatrixXd f(const Eigen::MatrixXd& x) const;
        int fSize() const;
        Eigen::MatrixXd xRvs(std::mt19937& mt) const;
        Eigen::MatrixXd getTheoretical() const;
    } modelSecondTerm;
    SE::MC estimatorSecondTerm;

    std::pair<int, std::vector<Eigen::MatrixXd>> sharedSamples;
    void initSharedSamples(int N, std::mt19937& mt);

   public:
    NMC(Model& model)
        : model{model},
          modelFirstTerm{model, sharedSamples},
          estimatorFirstTerm(modelFirstTerm),
          modelSecondTerm{model, sharedSamples},
          estimatorSecondTerm(modelSecondTerm) {}
    std::pair<int, Eigen::MatrixXd> estimate(int N, std::mt19937& mt);
    Eigen::MatrixXd getTheoretical() const;
};

class Smoothing : public ::Estimator {
   private:
    Model& model;
    NE::Smoothing& estimatorFirstTerm;
    SE::MC& estimatorSecondTerm;
    std::pair<int, std::vector<std::vector<Eigen::MatrixXd>>> sharedSamples;
    void initSharedSamples(int N, std::mt19937& mt);

   protected:
    class modelFirstTerm : public NE::Model {
       private:
        EVSI::Model& modelEVSI;
        std::pair<int, std::vector<std::vector<Eigen::MatrixXd>>>&
            sharedSamples;

       public:
        modelFirstTerm(
            EVSI::Model& model,
            std::pair<int, std::vector<std::vector<Eigen::MatrixXd>>>&
                sharedSamples)
            : Model{}, modelEVSI{model}, sharedSamples{sharedSamples} {};
        Eigen::MatrixXd f(const Eigen::MatrixXd& gy) const;
        Eigen::MatrixXd g(const std::vector<Eigen::MatrixXd>& yz) const;
        int fSize() const;
        int gSize() const;
        Eigen::MatrixXd yRvs(std::mt19937& mt) const;
        Eigen::MatrixXd zRvs(const Eigen::MatrixXd& y, std::mt19937& mt) const;
        std::vector<Eigen::MatrixXd> psaRvs(std::mt19937& mt) const;
        int ySize() const;
        int zSize() const;
        Eigen::MatrixXd getTheoretical() const;
    } modelFirstTerm;
    class modelSecondTerm : public SE::Model {
       private:
        EVSI::Model& modelEVSI;
        std::pair<int, std::vector<std::vector<Eigen::MatrixXd>>>&
            sharedSamples;

       public:
        modelSecondTerm(
            EVSI::Model& model,
            std::pair<int, std::vector<std::vector<Eigen::MatrixXd>>>&
                sharedSamples)
            : modelEVSI{model}, sharedSamples{sharedSamples} {};
        Eigen::MatrixXd f(const Eigen::MatrixXd& x) const;
        int fSize() const;
        Eigen::MatrixXd xRvs(std::mt19937& mt) const;
        Eigen::MatrixXd getTheoretical() const;
    } modelSecondTerm;

   public:
    Smoothing(Model& model,
              NE::Smoothing& estimatorFirstTerm,
              SE::MC& estimatorSecondTerm)
        : model{model},
          estimatorFirstTerm(estimatorFirstTerm),
          estimatorSecondTerm(estimatorSecondTerm),
          modelFirstTerm{model, sharedSamples},
          modelSecondTerm{model, sharedSamples} {}
    std::pair<int, Eigen::MatrixXd> estimate(int N, std::mt19937& mt);
    Eigen::MatrixXd getTheoretical() const;
};

class Proposed : public Smoothing {
   private:
    NE::Proposed estimatorFirstTerm;
    SE::MC estimatorSecondTerm;

   public:
    Proposed(Model& model)
        : Smoothing{model, estimatorFirstTerm, estimatorSecondTerm},
          estimatorFirstTerm(modelFirstTerm),
          estimatorSecondTerm(modelSecondTerm) {}
};

class ProposedWithRegression : public Smoothing {
   private:
    NE::ProposedWithRegression estimatorFirstTerm;
    SE::MC estimatorSecondTerm;

   public:
    ProposedWithRegression(Model& model)
        : Smoothing{model, estimatorFirstTerm, estimatorSecondTerm},
          estimatorFirstTerm(modelFirstTerm),
          estimatorSecondTerm(modelSecondTerm) {}
};

class Strong : public Smoothing {
   private:
    NE::Strong estimatorFirstTerm;
    SE::MC estimatorSecondTerm;

   public:
    Strong(Model& model, std::string tmpPath)
        : Smoothing{model, estimatorFirstTerm, estimatorSecondTerm},
          estimatorFirstTerm(modelFirstTerm, tmpPath),
          estimatorSecondTerm(modelSecondTerm) {}
};

}  // namespace EVSI

#include "methods/evsi/nmc.cpp"
#include "methods/evsi/smoothing.cpp"
#include "methods/ne/nmc.cpp"
#include "methods/ne/proposed.cpp"
#include "methods/ne/smoothing.cpp"
#include "methods/ne/strong.cpp"
#include "methods/se/mc.cpp"
#include "models/evsi/medical.cpp"
#include "models/evsi/testcase.cpp"
#include "models/ne/testcase.cpp"
#include "visualize.cpp"

#endif
