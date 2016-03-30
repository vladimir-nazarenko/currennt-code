#include "Optimizer.hpp"
#include "../helpers/Matrix.hpp"

namespace optimizers {

template <typename TDevice>
class Lbfgs : public Optimizer<TDevice> {

    typedef typename TDevice::real_vector real_vector;

    // Optimizer interface
protected:
    void _updateWeights();
public:
    Lbfgs(
            NeuralNetwork<TDevice> &neuralNetwork,
            data_sets::DataSet     &trainingSet,
            data_sets::DataSet     &validationSet,
            data_sets::DataSet     &testSet,
            int maxEpochs,
            int maxEpochsNoBest,
            int validateEvery,
            int testEvery
            );

    ~Lbfgs();
    void exportState(const helpers::JsonDocument &jsonDoc) const;
    void importState(const helpers::JsonDocument &jsonDoc);

private:
    void _writeWeights();
    void _readDerivatives(real_vector &output);
    real_vector mInversedHessian;
    real_vector mUpdateDirection;
    real_vector mDerivatives;
    real_vector mWeights;
    helpers::Matrix<TDevice> mInversedHessianMatrix;
    helpers::Matrix<TDevice> mUpdateDirectionMatrix;
    helpers::Matrix<TDevice> mWeigthsMatrix;
    helpers::Matrix<TDevice> mGradMatrix;
    int mNumberOfWeights;
};
}
