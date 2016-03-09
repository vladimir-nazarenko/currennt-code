#include <gtest/gtest.h>
//#include "../src/layers/LstmLayer.hpp"
#include "../src/helpers/Matrix.hpp"
#include "../src/Types.hpp"

using namespace helpers;

//class MatrixTest<TDevice> : public ::testing::Test {
//    typedef ::testing::Types<Cpu, Gpu> MyTypes;
//private:
//    Matrix<TypeParam> m1;
//public:
//    MatrixTest() {
//        TDevice::int_vector vec(4, 1);
////        m1 = Matrix<Cpu>(&vec, 2, 2);
//    }

//    virtual ~MatrixTest() {

//    }

//    int r5() {
//        return 5;
//    }
//};

//TYPED_TEST(Matrix, multiplicate) {
//    Ma
//}

//template <class TDevice>
//struct weight_matrices_t {
//    helpers::Matrix<TDevice> niInput;
//    helpers::Matrix<TDevice> igInput;
//    helpers::Matrix<TDevice> fgInput;
//    helpers::Matrix<TDevice> ogInput;
//    helpers::Matrix<TDevice> niInternal;
//    helpers::Matrix<TDevice> igInternal;
//    helpers::Matrix<TDevice> fgInternal;
//    helpers::Matrix<TDevice> ogInternal;
//};

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
//    Matrix<Gpu> m;
//    weight_matrices_t<Gpu> wm;
    return RUN_ALL_TESTS();
}
