#include "lbfgs.hpp"
#include "../helpers/Matrix.hpp"
#include "../helpers/getRawPointer.cuh"
#include <algorithm>

namespace internal {
namespace {

/* Recieves hessian guess, weights, weight updates and deltas
 * makes next guess for hessian, makes one iteration of bfgs
 */
struct UpdateWeightFn {

    real_t *inversedhessian;
    const real_t *weights;
    const real_t *weightUpdates;
    real_t *weightDeltas;
    const int weightsNumber;

    __host__ __device__ real_t operator() (const int &weightIdx) {
        real_t updateDirection = 0;
        for (int i = weightsNumber * weightIdx; i < weightsNumber * (weightIdx + 1); ++i) {
            updateDirection -= inversedhessian[i] * weightUpdates[i];
        }
        // lune search instead of constant
        real_t alpha = 0.5;
        real_t delta = alpha * updateDirection;
        weightDeltas[weightIdx] = delta;
        real_t newWeight = weights[weightIdx] + delta;



        return newWeight;
    }
};

struct CrossProductFn {
    real_t *y;
    real_t *s;
    real_t r;
    int dim;

    __host__ __device__ real_t operator() (const int &idx) {
        int row = idx / dim;
        int col = idx % dim;

        return y[row] * s[col] * r;
    }
};

}
}

namespace optimizers {

template <typename TDevice>
Lbfgs<TDevice>::Lbfgs(NeuralNetwork<TDevice> &neuralNetwork, data_sets::DataSet &trainingSet, data_sets::DataSet &validationSet, data_sets::DataSet &testSet, int maxEpochs, int maxEpochsNoBest, int validateEvery, int testEvery)
    : Optimizer<TDevice>(neuralNetwork, trainingSet, validationSet, testSet, maxEpochs, maxEpochsNoBest, validateEvery, testEvery)
{
    // compute the total number of weights
    for (int i = 1; i < this->_neuralNetwork().layers().size() - 1; ++i) {
        layers::TrainableLayer<TDevice> *layer = dynamic_cast<layers::TrainableLayer<TDevice>*>(this->_neuralNetwork().layers()[i].get());
        if(!layer)
            continue;

        mNumberOfWeights += layer->size();
    }

    // initialize vectors for matrices
    mDerivatives = Cpu::real_vector(this->mNumberOfWeights, 0);
    mWeights = Cpu::real_vector(this->mNumberOfWeights, 0);
    mDerivatives = Cpu::real_vector(this->mNumberOfWeights, 0);


    // initialize inversed hessian with identity matrix
    mInversedHessian = Cpu::real_vector(mNumberOfWeights * mNumberOfWeights, 0);
    for (int i = 0; i < mNumberOfWeights; ++i) {
        mInversedHessian[i * mNumberOfWeights] = 1;
    }

    // initialize matrices
    mInversedHessianMatrix = helpers::Matrix<TDevice>(&mInversedHessian, mNumberOfWeights, mNumberOfWeights);
    mGradMatrix = helpers::Matrix<TDevice>(&mDerivatives, mNumberOfWeights, 1);
    mUpdateDirectionMatrix = helpers::Matrix<TDevice>(&mUpdateDirection, mNumberOfWeights, 1);
    mWeigthsMatrix = helpers::Matrix<TDevice>(&mWeights, mNumberOfWeights, 1);
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

        thrust::copy_n(input.begin() + offset, layer->size(), layer->weights().begin());
        offset += layer->size();
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

        thrust::copy(this->_curWeightUpdates()[i].begin(), this->_curWeightUpdates()[i].end(), output.begin() + offset);
        offset += layer->size();
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

        thrust::copy(this->_curWeightUpdates()[i].begin(), this->_curWeightUpdates()[i].end(), mDerivatives.begin() + offset);
        thrust::copy(layer->weights().begin(), layer->weights().end(), mWeights.begin() + offset);
        offset += layer->size();
    }


    // d_k = -H_k * g_k
    mUpdateDirectionMatrix.assignProduct(mInversedHessianMatrix, false, mGradMatrix, false);
    thrust::transform(mUpdateDirection.begin(), mUpdateDirection.end(), mUpdateDirection.begin(), thrust::negate<float>());

    // line search
    real_t errorFnValue = this->_curTrainError();
    real_t alpha = 2;
    real_t beta = 0.5;
    real_t beta1 = 0.25;
    real_vector newGrad = mDerivatives;
    real_t newError;
    real_t gByD;
    real_t newGradByD;
    // suppose that if alpha = 1 doesn't satisfy Wolfe conditions, we use it nevertheless
    do {
        // start with alpha = 1
        alpha /= 2;

        // tmp = x_k + alpha_k * d_k
        real_vector tmp = mUpdateDirection;
        thrust::transform(tmp.begin(), tmp.end(), thrust::make_constant_iterator(alpha), tmp.begin(), thrust::multiplies<float>());
        thrust::transform(mWeights.begin(), mWeights.end(), tmp.begin(), tmp.begin(), thrust::plus<float>());

        // gByD = g_k' * d_k
        real_vector gByDvec = mUpdateDirection;
        thrust::transform(gByDvec.begin(), gByDvec.end(), mDerivatives.begin(), gByDvec.begin(), thrust::multiplies<float>());
        gByD = thrust::reduce(gByDvec.begin(), gByDvec.end());

        // update weights of neural network
        _writeWeights(tmp);

        // newGrad = g(x_k + alpha_k * d_k)
        newError = this->_computeForwardBackwardPassOnTrainset();
        _readDerivatives(newGrad);

        // newGradByD = g(x_k + alpha_k * d_k)' * d_k
        thrust::transform(newGrad.begin(), newGrad.end(), mUpdateDirection.begin(), newGrad.begin(), thrust::multiplies<float>());
        newGradByD = thrust::reduce(newGrad.begin(), newGrad.end());
        // check the Wolfe conditions
    } while (newError <= errorFnValue + beta1 * alpha * gByD &&
             newGradByD >= beta * gByD);

    // s = x_k+1 - x_k; y = g_k+1 - g_k
    real_vector s = mUpdateDirection;
    thrust::transform(s.begin(), s.end(), thrust::make_constant_iterator(std::min(alpha, (float)1)), s.begin(), thrust::multiplies<float>());

    // update the weights of neural network
    thrust::transform(s.begin(), s.end(), mWeights.begin(), mWeights.begin(), thrust::plus<float>());
    _writeWeights(mWeights);

    // update the inversed hessian
    real_vector y = newGrad;
    thrust::transform(y.begin(), y.end(), mDerivatives.begin(), y.begin(), thrust::minus<float>());

    real_vector v = mInversedHessian;
    real_vector sByS = mInversedHessian;

    internal::CrossProductFn getCross;
    getCross.dim = this->mNumberOfWeights;
    getCross.s = helpers::getRawPointer(s);
    getCross.y = helpers::getRawPointer(y);
    real_vector sByY = s;
    thrust::transform(s.begin(), s.end(), y.begin(), sByY.begin(), thrust::multiplies<float>());
    getCross.r = thrust::reduce(sByY.begin(), sByY.end());

    thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + mNumberOfWeights * mNumberOfWeights,
                      v.begin(), getCross);

    // I - ...
    thrust::transform(v.begin(), v.end(), v.begin(), thrust::negate<float>());
    for (int i = 0; i < mNumberOfWeights; ++i) {
        v[i * mNumberOfWeights] += 1;
    }

    getCross.y = getCross.s;

    thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0) + mNumberOfWeights * mNumberOfWeights,
                      sByS.begin(), getCross);

    helpers::Matrix<TDevice> vMatrix(&v, mNumberOfWeights, mNumberOfWeights);

    real_vector newInversedHessian(mNumberOfWeights * mNumberOfWeights, 0);

    helpers::Matrix<TDevice> newInversedHessianMatrix(&newInversedHessian, mNumberOfWeights, mNumberOfWeights);

    newInversedHessianMatrix.assignProduct(vMatrix, true, mInversedHessianMatrix, false);
    newInversedHessian.swap(mInversedHessian);
    newInversedHessianMatrix.assignProduct(mInversedHessianMatrix, false, vMatrix, false);
    newInversedHessian.swap(mInversedHessian);
    thrust::transform(mInversedHessian.begin(), mInversedHessian.end(), sByS.begin(), mInversedHessian.begin(), thrust::plus<float>());
}

template class Lbfgs<Cpu>;
template class Lbfgs<Gpu>;
}
