#include <random>

#include "all.cpp"

std::vector<int> Ns = {100, 1000, 10000, 100000, 1000000};
int R = 100;

int main() {
    std::mt19937 mt(1234);

    NE::TestCase m1;
    NE::NMC e11(m1);
    NE::Proposed e12(m1);
    NE::ProposedWithRegression e13(m1);
    NE::Strong e14(m1, "/dev/shm");
    printf("Estimating Nested Expectation with NMC for TestCase...\n");
    visualize(Ns, R, e11, "output/ne_testcase_nmc.txt", mt);
    printf(
        "Estimating Nested Expectation with Proposed Method for TestCase...\n");
    visualize(Ns, R, e12, "output/ne_testcase_proposed.txt", mt);
    printf(
        "Estimating Nested Expectation with Proposed Method with regression "
        "for TestCase...\n");
    visualize(Ns, R, e13, "output/ne_testcase_proposed_reg.txt", mt);
    printf(
        "Estimating Nested Expectation with Strong et al. for TestCase...\n");
    visualize(Ns, R, e14, "output/ne_testcase_strong.txt", mt);

    EVSI::TestCase m2(3);
    EVSI::NMC e21(m2);
    EVSI::Proposed e22(m2);
    EVSI::ProposedWithRegression e23(m2);
    EVSI::Strong e24(m2, "/dev/shm");
    printf("Estimating EVSI with NMC for TestCase...\n");
    visualize(Ns, R, e21, "output/evsi_testcase_nmc.txt", mt);
    printf("Estimating EVSI with Proposed Method for TestCase...\n");
    visualize(Ns, R, e22, "output/evsi_testcase_proposed.txt", mt);
    printf(
        "Estimating EVSI with Proposed Method with regression for "
        "TestCase...\n");
    visualize(Ns, R, e23, "output/evsi_testcase_proposed_reg.txt", mt);
    printf("Estimating EVSI with Strong et al. for TestCase...\n");
    visualize(Ns, R, e24, "output/evsi_testcase_strong.txt", mt);

    EVSI::Medical m3;
    EVSI::Proposed e32(m3);
    EVSI::ProposedWithRegression e33(m3);
    EVSI::Strong e34(m3, "/dev/shm");
    printf(
        "Estimating EVSI with Proposed Method for Medical Decision Model...\n");
    visualize(Ns, R, e32, "output/evsi_medical_proposed.txt", mt);
    printf(
        "Estimating EVSI with Proposed Method with regression for Medical "
        "Decision Model...\n");
    visualize(Ns, R, e33, "output/evsi_medical_proposed_reg.txt", mt);
    printf(
        "Estimating EVSI with Strong et al. for Medical Decision Model...\n");
    visualize(Ns, R, e34, "output/evsi_medical_strong.txt", mt);

    return 0;
}
