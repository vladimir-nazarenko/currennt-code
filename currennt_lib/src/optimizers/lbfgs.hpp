#include "Optimizer.hpp"

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
            int testEvery,
            real_t learningRate,
            int storageSize,
            real_t wolfeStepCoeff,
            real_t wolfeGradCoeff,
            real_t lineSearchStep
            );

    ~Lbfgs();
    void exportState(const helpers::JsonDocument &jsonDoc) const;
    void importState(const helpers::JsonDocument &jsonDoc);

private:
    void _writeWeights(real_vector &input);
    void _readDerivatives(real_vector &output);
    real_vector mUpdateDirection;
    real_vector mDerivatives;
    real_vector mWeights;
    int mNumberOfWeights;
    int m_rememberLast;
    // first element of the pair is previous update
    // second element of the pair is previous gradient difference
    // most recent pair is first
    std::vector<std::pair<real_vector, real_vector> > storage;
    real_t m_learnRate;
    real_t m_wolfeStepCoeff;
    real_t m_wolfeGradCoeff;
    real_t m_lineSearchStep;
};
}
