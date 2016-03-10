#include <gtest/gtest.h>
#include "../src/helpers/Matrix.hpp"
#include "../src/Types.hpp"

using namespace helpers;


TEST(Matrix, multiplication) {
    int N = 10;
    Cpu::real_vector vec(N, 1);
    for(int i = 0; i < N; ++i)
        vec[i] = i;
    Cpu::real_vector vec1(N, 1);
    for(int i = 0; i < N; ++i)
        vec1[i] = i;
    Cpu::real_vector vec2(4, 1);
    Matrix<Cpu> m(&vec, 2, 5);
    Matrix<Cpu> m1(&vec1, 5, 2);
    Matrix<Cpu> res(&vec2, 2, 2);
    res.assignProduct(m, false, m1, false);
    float exp[] = {60, 70, 160, 195};
    std::vector<float> expected(exp, exp + sizeof(exp) * sizeof(float));
    ASSERT_TRUE(std::equal(vec2.begin(), vec2.end(), expected.begin()));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
