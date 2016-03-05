#include "RNNLayer.hpp"
#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../helpers/getRawPointer.cuh"


namespace internal {
namespace {
    typedef activation_functions::Tanh output_act_fn_t;
    typedef activation_functions::Logistic input_act_fn_t;

    template <typename TActFn>
    struct ComputeOutputFn {
        const real_t *inputActs;
        const real_t *recurrentActs;
        const real_t *inputWeights;
        const real_t *recurrentWeights;
        const real_t *outputWeights;
        int layerSize;

        __host__ __device__ real_t operator() (const int &outputIdx) {
            real_t hidden[layerSize];
//            real_t hidden[layerSize];
            for(int i = 0; i < layerSize; ++i) {
                hidden[i] = input_act_fn_t::fn(
                            inputActs[i] * inputWeights[i] +
                            recurrentActs[i] * recurrentWeights[i]
                            );

            }


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
    return m_isBidirectional ? std::string("brnn") : std::string("rnn");
}


template <typename TDevice>
void RNNLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
{
    TrainableLayer<TDevice>::loadSequences(fraction);

    // Don't understand how this thing is supposed to work
    // Previous layer should not be aware that we train our network
    // on sequences. Or maybe it is?
    m_precedingLayerOutputMatrix = helpers::Matrix<TDevice>(&precedingLayer().outputs(), precedingLayer().size(), curMaxSeqLength() * parallelSequences());

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
    // sum up the activations from the preceding layer
    {{
        // forward states
        m_fw.igActsMatrix.assignProduct(m_fw.weightMatrices.igInput, true, m_precLayerOutputsMatrix, false);

        // backward states
        if (m_isBidirectional) {
            m_bw.igActsMatrix.assignProduct(m_bw.weightMatrices.igInput, true, m_precLayerOutputsMatrix, false);
        }
    }}

    // compute the block outputs
    {{
        int els = this->size() / (m_isBidirectional ? 2 : 1);
        int n   = this->parallelSequences() * els;

        // forward states
        internal::ComputeBlockOutputFn fn;
        fn.effLayerSize       = els;
        fn.prevOutputDistance = -n;
        fn.igActs             = helpers::getRawPointer(m_fw.igActs);

        for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
            // collect outputs from previous timestep
            if (timestep != 0) {
                m_fw.timestepMatrices[timestep].igActs.addProduct(m_fw.weightMatrices.igInternal, true, m_fw.timestepMatrices[timestep-1].hiddenTmpOutputs, false);
            }

            // compute outputs
            thrust::transform(
                thrust::counting_iterator<int>(n*timestep),
                thrust::counting_iterator<int>(n*timestep) + n,
                thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<bool>(!timestep),
                                                             thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                m_fw.tmpOutputs.begin() + n*timestep,
                fn
                );
        }

        }}

        /*
        // backward states
        if (m_isBidirectional) {
            fn.prevOutputDistance = +n;
            fn.niBiasWeights     += els;
            fn.igBiasWeights     += els;
            fn.fgBiasWeights     += els;
            fn.ogBiasWeights     += els;
            fn.igPeepWeights     += els;
            fn.fgPeepWeights     += els;
            fn.ogPeepWeights     += els;
            fn.cellStates         = helpers::getRawPointer(m_bw.cellStates);
            fn.niActs             = helpers::getRawPointer(m_bw.niActs);
            fn.igActs             = helpers::getRawPointer(m_bw.igActs);
            fn.fgActs             = helpers::getRawPointer(m_bw.fgActs);
            fn.ogActs             = helpers::getRawPointer(m_bw.ogActs);

            for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
                // collect outputs from previous timestep
                if (timestep != this->curMaxSeqLength()-1) {
                    m_bw.timestepMatrices[timestep].niActs.addProduct(m_bw.weightMatrices.niInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                    m_bw.timestepMatrices[timestep].igActs.addProduct(m_bw.weightMatrices.igInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                    m_bw.timestepMatrices[timestep].fgActs.addProduct(m_bw.weightMatrices.fgInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                    m_bw.timestepMatrices[timestep].ogActs.addProduct(m_bw.weightMatrices.ogInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                }

                // compute outputs
                thrust::transform(
                    thrust::counting_iterator<int>(n*timestep),
                    thrust::counting_iterator<int>(n*timestep) + n,
                    thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1), thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                    m_bw.tmpOutputs.begin() + n*timestep,
                    fn
                    );
            }
        }
    }}

    // resort outputs
    if (m_isBidirectional) {
        internal::ResortOutputsFn fn;
        fn.layerSize    = this->size();
        fn.effLayerSize = this->size() / 2;
        fn.fwOutputs    = helpers::getRawPointer(m_fw.tmpOutputs);
        fn.bwOutputs    = helpers::getRawPointer(m_bw.tmpOutputs);

        // gets elements from the first iterator (params 1 and 2),
        // applies the fn and writes the result to the second one
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(0) + this->curMaxSeqLength() * this->parallelSequences() * this->size(),
            this->_outputs().begin(),
            fn
            );
    }
    else {
        this->_outputs().swap(m_fw.tmpOutputs);
    }
    */
}
