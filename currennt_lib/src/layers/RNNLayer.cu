#include "RNNLayer.hpp"
#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../helpers/getRawPointer.cuh"
#include <thrust/iterator/constant_iterator.h>
//#include <thrust/random.h>


namespace internal {
namespace {
    typedef activation_functions::Tanh output_act_fn_t;
    typedef activation_functions::Logistic input_act_fn_t;

    // this functor gets the value and index and stores the value in one of vectors for errors:
    // one for forward state and another for backward according to the following layout.
    // The values of the forward and backward states are mingled so the second timestep of forward errors
    // goes after the first timestep of backward errors.
    struct ResortOutputErrorsFn
    {
        int layerSize;
        int effLayerSize;

        real_t *fwOutputErrors;
        real_t *bwOutputErrors;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
        {
            // unpack the tuple
            real_t outputErr = t.get<0>();
            int    outputIdx = t.get<1>();

            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                fwOutputErrors[offset] = outputErr;
            else
                bwOutputErrors[offset - effLayerSize] = outputErr;
        }
    };

    struct ComputeBlockHiddenFn {
        const real_t *recActs;
//        thrust::default_random_engine eng;
//        thrust::normal_distribution<float> dist;

        __host__ __device__ real_t operator() (const int &outputIdx, const bool &isFirstTimestep) {
            return isFirstTimestep ?
                        0 :
                        input_act_fn_t::fn(recActs[outputIdx]);
        }
    };

    struct ComputeBlockOutputFn {
        const real_t *outActs;

        __host__ __device__ real_t operator() (const int &outputIdx) {
            return output_act_fn_t::fn(outActs[outputIdx]);
        }
    };
}
}

namespace layers {

template <typename TDevice>
RNNLayer<TDevice>::RNNLayer(const helpers::JsonValue &layerChild, const helpers::JsonValue &weightsSection, Layer<TDevice> &precedingLayer, bool bidirectional)
    : TrainableLayer<TDevice>(layerChild, weightsSection, 4/*check*/, 4, precedingLayer),
      m_isBidirectional(bidirectional)
{
    forward_backward_info_t *fwbwArr[] = { &m_fw, &m_bw };
    for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
        forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

        // calculate sizes
        // previous layer size as defined by user
        int pls = this->precedingLayer().size();
        // current layer size as defined by user
        // for bidirectional layer it's 2xsize
        int ls  = this->size();
        int els = this->size() / (m_isBidirectional ? 2 : 1);

        Cpu::real_vector tmp(this->outputs().size() / (m_isBidirectional ? 2 : 1), 0);

        // call copy constructor to initialize vectors
        fwbw->tmpOutputs       = tmp;
        fwbw->tmpOutputErrors  = tmp;
        fwbw->hiddenTmpErrors  = tmp;
        fwbw->hiddenTmpOutputs = tmp;
        fwbw->igActs           = tmp;
        fwbw->igDeltas         = tmp;
        fwbw->igActs           = tmp;
        fwbw->ogActs           = tmp;
        fwbw->ogDeltas         = tmp;

        weight_matrices_t *wmArr[] = { &fwbw->weightMatrices, &fwbw->weightUpdateMatrices };
        real_vector*       wtsArr[] = { &this->weights(),      &this->_weightUpdates() };
        for (int wmArrIdx = 0; wmArrIdx < 2; ++wmArrIdx) {
            weight_matrices_t *wm  = wmArr [wmArrIdx];
            real_vector       *wts = wtsArr[wmArrIdx];

            int numInputWeights      = ls * pls;
            int numInternalWeights   = ls * els;
            int inputWeightsStart    = ((fwbwArrIdx == 1) ? (numInputWeights    / 2) : 0);
            int internalWeightsStart = ((fwbwArrIdx == 1) ? (numInternalWeights / 2) : 0) + 4 * (ls * (pls + 1));

            wm->igInput = helpers::Matrix<TDevice>(wts, pls, els, inputWeightsStart + 0 * numInputWeights);

            wm->igInternal = helpers::Matrix<TDevice>(wts, els, els, internalWeightsStart + 0 * numInternalWeights);
            wm->ogInternal = helpers::Matrix<TDevice>(wts, els, els, internalWeightsStart + 1 * numInternalWeights);
        }

        // matrices for each timestep
        for (int timestep = 0; timestep < this->maxSeqLength(); ++timestep) {
            int rows   = this->size() / (m_isBidirectional ? 2 : 1);
            int cols   = this->parallelSequences();
            int offset = timestep * rows * cols;

            timestep_matrices_t tm;
            tm.tmpOutputs       = helpers::Matrix<TDevice>(&fwbw->tmpOutputs,      rows, cols, offset);
            tm.tmpOutputErrors  = helpers::Matrix<TDevice>(&fwbw->tmpOutputErrors, rows, cols, offset);
            tm.hiddenTmpErrors  = helpers::Matrix<TDevice>(&fwbw->hiddenTmpErrors, rows, cols, offset);
            tm.hiddenTmpOutputs = helpers::Matrix<TDevice>(&fwbw->hiddenTmpOutputs,rows, cols, offset);
            tm.igActs           = helpers::Matrix<TDevice>(&fwbw->igActs,          rows, cols, offset);
            tm.ogActs           = helpers::Matrix<TDevice>(&fwbw->ogActs,          rows, cols, offset);
            tm.igDeltas         = helpers::Matrix<TDevice>(&fwbw->igDeltas,        rows, cols, offset);
            tm.ogDeltas         = helpers::Matrix<TDevice>(&fwbw->ogDeltas,        rows, cols, offset);

            fwbw->timestepMatrices.push_back(tm);
        }

    }
}

template <typename TDevice>
const std::string &RNNLayer<TDevice>::type() const
{
    const std::string bi("rnn");
    const std::string un("brnn");
    return m_isBidirectional ? bi : un;
}


template <typename TDevice>
void RNNLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
{
    TrainableLayer<TDevice>::loadSequences(fraction);

    // Don't understand how this thing is supposed to work
    // Previous layer should not be aware that we train our network
    // on sequences. Or maybe it is?
    m_precedingLayerOutputMatrix = helpers::Matrix<TDevice>(&this->precedingLayer().outputs(), this->precedingLayer().size(), this->curMaxSeqLength() * this->parallelSequences());

    forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
    for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
        forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

        int rows = this->size() / (m_isBidirectional ? 2 : 1);
        int cols = this->curMaxSeqLength() * this->parallelSequences();

        fwbw->igActsMatrix = helpers::Matrix<TDevice>(&fwbw->igActs, rows, cols);
        fwbw->ogActsMatrix = helpers::Matrix<TDevice>(&fwbw->ogActs, rows, cols);

        fwbw->igDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->igDeltas, rows, cols);
        fwbw->ogDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->ogDeltas, rows, cols);
    }
}

template <typename TDevice>
void RNNLayer<TDevice>::computeForwardPass()
{
    // TODO: Check implementation

    // sum up the activations from the preceding layer
    // forward states
    m_fw.igActsMatrix.assignProduct(m_fw.weightMatrices.igInput, true, m_precedingLayerOutputMatrix, false);

    // compute the block outputs
    {{
        int els = this->size() / (m_isBidirectional ? 2 : 1);
        int n   = this->parallelSequences() * els;

        // compute internal states
        internal::ComputeBlockHiddenFn fn;
        fn.recActs = helpers::getRawPointer(m_fw.hiddenTmpOutputs);
//        thrust::normal_distribution<float> dist(0, 0.0001);
//        fn.dist(dist);

        for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
            // collect outputs from previous timestep
            if (timestep != 0) {
                m_fw.timestepMatrices[timestep].igActs.addProduct(m_fw.weightMatrices.igInternal, true, m_fw.timestepMatrices[timestep-1].hiddenTmpOutputs, false);
            }

            // compute outputs
            thrust::transform(
                thrust::counting_iterator<int>(n*timestep),
                thrust::counting_iterator<int>(n*timestep) + n,
                thrust::constant_iterator<bool>(!timestep),
                m_fw.hiddenTmpOutputs.begin() + n*timestep,
                fn
                );
        }

        // compute outputs
        m_fw.ogActsMatrix.assignProduct(m_fw.weightMatrices.ogInternal, true, m_fw.hiddenTmpOutputsMatrix, false);
        internal::ComputeBlockOutputFn fnOut;
        fnOut.outActs = helpers::getRawPointer(m_fw.ogActs);
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>((n + 1) * this->curMaxSeqLength()),
            m_fw.tmpOutputs.begin(),
            fnOut
            );

    }}

    // backward states
    if (m_isBidirectional) {
        m_bw.igActsMatrix.assignProduct(m_bw.weightMatrices.igInput, true, m_precedingLayerOutputMatrix, false);


        // compute the block outputs
        int els = this->size() / (m_isBidirectional ? 2 : 1);
        int n   = this->parallelSequences() * els;

        // compute internal states
        internal::ComputeBlockHiddenFn fn;
        fn.recActs = helpers::getRawPointer(m_bw.hiddenTmpOutputs);
//        thrust::normal_distribution<float> dist(0, 0.0001);
//        fn.dist(dist);

        for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
            // collect outputs from previous timestep
            if (timestep != this->curMaxSeqLength()-1) {
                m_bw.timestepMatrices[timestep].igActs.addProduct(m_bw.weightMatrices.igInternal, true, m_bw.timestepMatrices[timestep-1].hiddenTmpOutputs, false);
            }

            // compute outputs
            thrust::transform(
                thrust::counting_iterator<int>(n*timestep),
                thrust::counting_iterator<int>(n*timestep) + n,
                thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1),
                m_bw.hiddenTmpOutputs.begin() + n*timestep,
                fn
                );
        }

        // compute outputs
        m_bw.ogActsMatrix.assignProduct(m_bw.weightMatrices.ogInternal, true, m_bw.hiddenTmpOutputsMatrix, false);
        internal::ComputeBlockOutputFn fnOut;
        fnOut.outActs = helpers::getRawPointer(m_bw.ogActs);
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>((n + 1) * this->curMaxSeqLength()),
            m_bw.tmpOutputs.begin(),
            fnOut
            );
        }
}

template <typename TDevice>
void RNNLayer<TDevice>::computeBackwardPass() {
    // for unidirectional LSTM, we can write the output errors directly in the layer output errors vector
    // CHECK: it's not clear if this action leads to correct results
    if (m_isBidirectional) {
        internal::ResortOutputErrorsFn fn;
        fn.layerSize      = this->size();
        fn.effLayerSize   = this->size() / 2;
        fn.fwOutputErrors = helpers::getRawPointer(m_fw.tmpOutputErrors);
        fn.bwOutputErrors = helpers::getRawPointer(m_bw.tmpOutputErrors);

        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin(),   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin()+n, thrust::counting_iterator<int>(0)+n)),
            fn
            );
    }
    else {
        m_fw.tmpOutputs     .swap(this->outputs());
        m_fw.tmpOutputErrors.swap(this->outputErrors());
    }


}

// explicit template instantiations
template class RNNLayer<Cpu>;
template class RNNLayer<Gpu>;

}
