#include "lbfgs.hpp"
#include "../helpers/getRawPointer.cuh"
#include <algorithm>
#include <thrust/inner_product.h>
#include <stack>
#include <boost/range/adaptor/reversed.hpp>
#include "../Configuration.hpp"

namespace optimizers {

template <typename TDevice>
Lbfgs<TDevice>::Lbfgs(NeuralNetwork<TDevice> &neuralNetwork, data_sets::DataSet &trainingSet, data_sets::DataSet &validationSet, data_sets::DataSet &testSet,
                      int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery, real_t learningRate,
                      int storageSize, real_t wolfeStepCoeff, real_t wolfeGradCoeff, real_t lineSearchStep, real_t learnRateForFirstIter)
    : Optimizer<TDevice>(neuralNetwork, trainingSet, validationSet, testSet, maxEpochs, maxEpochsNoBest, validateEvery, testEvery),
      m_learnRate      (learningRate),
      m_rememberLast   (storageSize),
      m_wolfeStepCoeff (wolfeStepCoeff),
      m_wolfeGradCoeff (wolfeGradCoeff),
      m_lineSearchStep (lineSearchStep),
      m_learnRateForFirstIter(learnRateForFirstIter)
{
    mNumberOfWeights = 0;
    // compute the total number of weights
    for (int i = 1; i < this->_neuralNetwork().layers().size() - 1; ++i) {
        layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
        if(!layer)
            continue;

        mNumberOfWeights += layer->weights().size();
    }

    // initialize vectors for matrices
    mDerivatives = real_vector(this->mNumberOfWeights, 0);
    mWeights = real_vector(this->mNumberOfWeights, 0);
    mUpdateDirection = real_vector(this->mNumberOfWeights, 0);
}

template <typename TDevice>
Lbfgs<TDevice>::~Lbfgs()
{

}

template <typename TDevice>
void Lbfgs<TDevice>::exportState(const helpers::JsonDocument &jsonDoc) const
{

}

template <typename TDevice>
void Lbfgs<TDevice>::importState(const helpers::JsonDocument &jsonDoc)
{

}

template <typename TDevice>
void Lbfgs<TDevice>::_writeWeights(real_vector &input)
{
    int offset = 0;
    for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
        layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
        if (!layer)
            continue;

        thrust::copy_n(input.begin() + offset, layer->weights().size(), layer->weights().begin());
        offset += layer->weights().size();
    }
}

template <typename TDevice>
void Lbfgs<TDevice>::_readDerivatives(real_vector &output)
{
    int offset = 0;
    for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
        layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
        if (!layer)
            continue;

        if(Configuration::instance().hybridOnlineBatch())
            thrust::copy(layer->weightUpdates().begin(), layer->weightUpdates().end(), output.begin() + offset);
        else
            thrust::copy(this->_curWeightUpdates()[i].begin(), this->_curWeightUpdates()[i].end(), output.begin() + offset);
        offset += layer->weights().size();
    }
}

template <typename TDevice>
void Lbfgs<TDevice>::_updateWeights()
{
    // load weights and derivatives
    int offset = 0;
    for (size_t i = 1; i < this->_neuralNetwork().layers().size()-1; ++i) {
        layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
        if (!layer)
            continue;
        thrust::copy(layer->weights().begin(), layer->weights().end(), mWeights.begin() + offset);
        offset += layer->weights().size();
    }
    _readDerivatives(mDerivatives);

    // compute lbfgs update
    std::stack<real_t> prev_updates_by_gradient;
    std::stack<real_t> gradientByUpdate;
    // q <- df/dw
    real_vector gradientUpdateAccumulator(mNumberOfWeights, 0);
    thrust::copy(mDerivatives.begin(), mDerivatives.end(), gradientUpdateAccumulator.begin());
    for(auto &pair : storage) {
        // 1 / rho_i <- s_i * y_i
        real_t inner_product_s_stored_grad = thrust::inner_product(pair.first.begin(), pair.first.end(),  pair.second.begin(), 0.0f);
        // s_i * q
        real_t inner_product_s_cur_grad    = thrust::inner_product(pair.first.begin(), pair.first.end(), gradientUpdateAccumulator.begin(), 0.0f);
        // alpha_i <- rho_i * s_i * q
        real_t upd_by_grad = (1 / inner_product_s_stored_grad) * inner_product_s_cur_grad;

        // push both values to stack to use in computing the hessian guess
        prev_updates_by_gradient.push(upd_by_grad);
        gradientByUpdate.push(inner_product_s_stored_grad);

        // tmp <- -alpha_i * y_i
        real_vector tmp(mNumberOfWeights, 0);
        thrust::transform(pair.second.begin(), pair.second.end(), thrust::make_constant_iterator(-upd_by_grad), tmp.begin(), thrust::multiplies<float>());

        // q <- q - alpha_i * y_i
        thrust::transform(gradientUpdateAccumulator.begin(), gradientUpdateAccumulator.end(), tmp.begin(), gradientUpdateAccumulator.begin(), thrust::plus<float>());
    }

    // given hessian initial guess is diagonal matrix
    real_vector hessian_guess_by_gradient(mNumberOfWeights, 0);
    real_t scale_factor = 1;
    // gamma_k = s_k-1 * y_k-1/y_k-1*y_k-1
    if(storage.size() > 0) {
        scale_factor = thrust::inner_product((*storage.begin()).first.begin(), (*storage.begin()).first.end(), (*storage.begin()).second.begin(), 0.0f) /
                thrust::inner_product((*storage.begin()).second.begin(), (*storage.begin()).second.end(), (*storage.begin()).second.begin(), 0.0f);
//    } else {
//        scale_factor = 1 /
//                std::fabs(thrust::inner_product(mDerivatives.begin(), mDerivatives.end(), mDerivatives.begin(), 0.0f));
    }
    // r <- H_K^0 * q
    thrust::transform(gradientUpdateAccumulator.begin(), gradientUpdateAccumulator.end(), thrust::make_constant_iterator(scale_factor),
                      hessian_guess_by_gradient.begin(), thrust::multiplies<float>());

    for(const auto & pair : boost::adaptors::reverse(storage)) {
        // beta <- rho_i*y_i*r
        real_t prevGradUpdateByCurrentGuess = (1 / gradientByUpdate.top()) *
                thrust::inner_product(pair.second.begin(), pair.second.end(), hessian_guess_by_gradient.begin(), 0.0f);
        gradientByUpdate.pop();

        // tmp <- s_i * (alpha_i - beta)
        real_vector tmp(mNumberOfWeights, 0);
        thrust::transform(pair.first.begin(), pair.first.end(),
                          thrust::make_constant_iterator(prev_updates_by_gradient.top() - prevGradUpdateByCurrentGuess), tmp.begin(), thrust::multiplies<float>());
        prev_updates_by_gradient.pop();

        // r <- r + tmp
        thrust::transform(hessian_guess_by_gradient.begin(), hessian_guess_by_gradient.end(), tmp.begin(), hessian_guess_by_gradient.begin(), thrust::plus<float>());
    }

    // d_k = -H_k * g_k
    thrust::transform(hessian_guess_by_gradient.begin(), hessian_guess_by_gradient.end(), mUpdateDirection.begin(), thrust::negate<float>());

    real_t curStep = m_learnRate * m_lineSearchStep;
    // line search
//    if(this->currentEpoch() == 1)
//        curStep = m_learnRateForFirstIter * m_lineSearchStep;
    real_t errorFnValue = this->_curTrainError();
    // multiplication is for the first step
    real_vector newGrad(mDerivatives.size(), 0);
    real_t newError = 0.0f;
    // gByD = g_k' * d_k
    real_t gByD = thrust::inner_product(mUpdateDirection.begin(), mUpdateDirection.end(), mDerivatives.begin(), 0.0f);
    real_t newGradByD = 0.0f;
    do {
        curStep /= m_lineSearchStep;

        // tmp = x_k + alpha_k * d_k
        real_vector tmp = mUpdateDirection;
        thrust::transform(tmp.begin(), tmp.end(), thrust::make_constant_iterator(curStep), tmp.begin(), thrust::multiplies<float>());
        thrust::transform(mWeights.begin(), mWeights.end(), tmp.begin(), tmp.begin(), thrust::plus<float>());

        // update weights of neural network
        _writeWeights(tmp);

        // newGrad = g(x_k + alpha_k * d_k)
        newError = this->_computeForwardBackwardPassOnTrainset();
        _readDerivatives(newGrad);

        // newGradByD = g(x_k + alpha_k * d_k)' * d_k
        newGradByD = thrust::inner_product(mUpdateDirection.begin(), mUpdateDirection.end(), newGrad.begin(), 0.0f);
        // check the Wolfe conditions

//        if(! (newError <= errorFnValue + m_wolfeStepCoeff * curStep * gByD)) {
//            std::cout << "STEP";
//        }
//        if(! (newGradByD >= m_wolfeGradCoeff * gByD)){
//            std::cout << "grad";
//        }


    } while (!(newError <= errorFnValue + m_wolfeStepCoeff * curStep * gByD &&
             newGradByD >= m_wolfeGradCoeff * gByD) && curStep > 1e-5);

    // s = x_k+1 - x_k; y = g_k+1 - g_k
    real_vector s(mNumberOfWeights, 0);
    thrust::transform(mUpdateDirection.begin(), mUpdateDirection.end(), thrust::make_constant_iterator(curStep), s.begin(), thrust::multiplies<float>());
    real_vector y(mNumberOfWeights, 0);
    thrust::transform(newGrad.begin(), newGrad.end(), mDerivatives.begin(), y.begin(), thrust::minus<float>());

//    m_learnRate = sqrt(curStep);


//    std::cout << "ALPHA:" << curStep << "\n";

    if(storage.size() == this->m_rememberLast && this->m_rememberLast > 0) {
        storage.pop_back();
    }

    if(m_rememberLast > 0)
        storage.insert(storage.begin(), std::make_pair(s, y));
}

template class Lbfgs<Cpu>;
template class Lbfgs<Gpu>;
}
