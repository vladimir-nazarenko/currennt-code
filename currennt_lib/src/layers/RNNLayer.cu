#include "RNNLayer.hpp"
#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/limitedError.cuh"
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

    /**
     * Gets two vectors with outputs from forward and backward directions and merges them into one vector
     * such that frames of the backward outputs alternate with the ones of the forward outputs. Forward frame comes first.
     */
    struct ResortOutputsFn
    {
        int layerSize;
        int effLayerSize;

        const real_t *fwOutputs;
        const real_t *bwOutputs;

        __host__ __device__ real_t operator() (const int &outputIdx) const
        {
            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                return fwOutputs[offset];
            else
                return bwOutputs[offset - effLayerSize];
        }
    };

    struct ComputeBlockOutputFn {
        int els;
        const real_t *igActs;
        real_t bias;
        const real_t *biasWeights;

        __host__ __device__ real_t operator() (const int &outputIdx, const bool &isFirstTimestep) {
            return isFirstTimestep ?
                        0 :
                        input_act_fn_t::fn(igActs[outputIdx] + bias * biasWeights[outputIdx % els]);
        }
    };

    /**
     * Computes the derivatives of the outputs
     * Not sure if it's crrect
     */
    struct ComputeBlockErrorsFn
    {
        int effLayerSize;
        int prevOutputDistance;

        const char *patTypes;
        const real_t *igActs;
        real_t *igDeltas;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int, bool, bool, bool> &t) const
        {
            // unpack the tuple
            real_t hiddenErr    = t.get<0>();
            int    outputIdx    = t.get<1>();
            bool   firstCall    = t.get<2>();
            bool   lastCall     = t.get<3>();
            bool   checkPatType = t.get<4>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    igDeltas       [outputIdx] = 0;
                    return;
                }
            }

            real_t igAct = igActs[outputIdx];

            real_t igDelta = input_act_fn_t::deriv(igAct) * hiddenErr;

            igDeltas[outputIdx] = helpers::limitedError(igDelta);
        }
    };

    struct ComputeWeightUpdateFn
    {
        int    layerSize;
        int    effLayerSize;
        int    precLayerSize;
        int    timestepDistance;
        int    parallelSequences;
        // pattern is the pair of (input vector, output vector) for one timestep
        int    patternsCount;
        int    biasWeightsOffset;
        int    internalWeightsOffset;
        real_t bias;

        const real_t *plOutputs;
        const real_t *fwOutputs;
        const real_t *bwOutputs;
        const real_t *fwIgDeltas;
        const real_t *bwIgDeltas;

        __host__ __device__ real_t operator() (const int &weightIdx) const
        {
            // determine the weight type
            //
            // weightType = 0bXX with XX = {input, bias, internal, peephole}
            // weightType = 0b00 ( 0): input weight
            //              0b01 ( 1): bias weight
            //              0b10 ( 2): internal weight
            int inwc = layerSize * precLayerSize;
            int biwc = layerSize;
            int itwc = layerSize * effLayerSize;

            int weightType = (int)(weightIdx >= 0                 + 1 * inwc) +
                             (int)(weightIdx >= biasWeightsOffset + 1 * biwc);

            // calculate indices, offsets and increments
            const real_t *offOutputs;
            int           tgtBlockIdx;
            int           offOutputsInc;
            bool          skipFirstPattern = false;
            bool          skipLastPattern  = false;
            bool          isBwStateWeight;

            switch (weightType) {
            // input weight
            case 0x0:
                {{
                    // calculate indices
                    int inputWeightIdx = weightIdx;
                    int plBlockIdx     = inputWeightIdx % precLayerSize;
                    int blockIdx       = inputWeightIdx / precLayerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = &plOutputs[plBlockIdx];
                    offOutputsInc = precLayerSize;
                }}
                break;

            // bias weight
            case 0x4:
                {{
                    // calculate indices
                    int biasWeightIdx = weightIdx - biasWeightsOffset;
                    int blockIdx      = biasWeightIdx;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = NULL;
                    offOutputsInc = 0;
                }}
                break;

            // internal weight
            case 0x8:
                {{
                    // calculate indices
                    int internalWeightIdx = weightIdx - internalWeightsOffset;
                    int srcBlockIdx       = internalWeightIdx % effLayerSize;
                    int blockIdx          = internalWeightIdx / effLayerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = (isBwStateWeight ? &bwOutputs[srcBlockIdx] : &fwOutputs[srcBlockIdx]);
                    offOutputsInc = effLayerSize;

                    if (isBwStateWeight) {
                        offOutputs += timestepDistance;
                        skipLastPattern = true;
                    }
                    else {
                        offOutputs -= timestepDistance;
                        skipFirstPattern = true;
                    }
                }}
                break;
            }

            // determine the start of the delta values
            const real_t *igDeltasLut[] = {
                fwIgDeltas,
                bwIgDeltas,
            };

            // calculate the weight update over all patterns

            const real_t *offDeltas = &igDeltasLut[isBwStateWeight ? 1 : 0][tgtBlockIdx];

            if (skipFirstPattern) {
                offOutputs += parallelSequences * offOutputsInc;
                offDeltas  += parallelSequences * effLayerSize;
            }

            int numPatterns = patternsCount;
            if (skipFirstPattern || skipLastPattern)
                numPatterns -= parallelSequences;

            real_t wu = 0;
            for (int i = 0; i < numPatterns; ++i) {
//                wu += (offOutputs ? *offOutputs : bias) * *offDeltas;

                offOutputs += offOutputsInc;
                offDeltas  += effLayerSize;
            }
            wu = *offDeltas;

            return wu;
        }
    };

}
}

namespace layers {

template <typename TDevice>
RNNLayer<TDevice>::RNNLayer(const helpers::JsonValue &layerChild, const helpers::JsonValue &weightsSection, Layer<TDevice> &precedingLayer, bool bidirectional)
    : TrainableLayer<TDevice>(layerChild, weightsSection, 1 /* ls biases stored here */
                                                        , helpers::safeJsonGetInt(layerChild, "size") / (bidirectional ? 1 : 2)
                                                        , precedingLayer),
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
        fwbw->hiddenTmpErrors  = tmp;
        fwbw->hiddenTmpOutputs = tmp;
        fwbw->igActs           = tmp;
        fwbw->igDeltas         = tmp;


        weight_matrices_t *wmArr[] = { &fwbw->weightMatrices, &fwbw->weightUpdateMatrices };
        real_vector*       wtsArr[] = { &this->weights(),      &this->_weightUpdates() };
        /** Weights layout
         * [inputWeightsForward][inputWeightsBackward][biasInputWeightsForward][biasInputWeightsBackward] ...
         *  ... [internalWeightsForward][internalWeightsBackward]
         */
        for (int wmArrIdx = 0; wmArrIdx < 2; ++wmArrIdx) {
            weight_matrices_t *wm  = wmArr [wmArrIdx];
            real_vector       *wts = wtsArr[wmArrIdx];

            int numInputWeights      = ls * pls;
            int numInternalWeights   = ls * els;
            int numBiasWeights       = ls;
            int inputWeightsStart    = ((fwbwArrIdx == 1) ? (numInputWeights    / 2) : 0);
            int internalWeightsStart = ((fwbwArrIdx == 1) ? (numInternalWeights / 2) : 0) +
                                       numInputWeights + numBiasWeights;

            wm->igInput     = helpers::Matrix<TDevice>(wts, pls, els, inputWeightsStart   );
            wm->igInternal  = helpers::Matrix<TDevice>(wts, els, els, internalWeightsStart);

        }

        _rawBiasWeights = helpers::getRawPointer(this->weights()) + ls * pls;

        // matrices for each timestep
        for (int timestep = 0; timestep < this->maxSeqLength(); ++timestep) {
            int rows   = this->size() / (m_isBidirectional ? 2 : 1);
            int cols   = this->parallelSequences();
            int offset = timestep * rows * cols;

            timestep_matrices_t tm;
            tm.hiddenTmpErrors  = helpers::Matrix<TDevice>(&fwbw->hiddenTmpErrors, rows, cols, offset);
            tm.hiddenTmpOutputs = helpers::Matrix<TDevice>(&fwbw->hiddenTmpOutputs,rows, cols, offset);
            tm.igActs           = helpers::Matrix<TDevice>(&fwbw->igActs,          rows, cols, offset);
            tm.igDeltas         = helpers::Matrix<TDevice>(&fwbw->igDeltas,        rows, cols, offset);

            fwbw->timestepMatrices.push_back(tm);
        }

    }
}

template <typename TDevice>
const std::string &RNNLayer<TDevice>::type() const
{
    static const std::string su("lrnn");
    static const std::string sb("brnn");
    return (m_isBidirectional ? sb : su);
}


template <typename TDevice>
void RNNLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
{
    TrainableLayer<TDevice>::loadSequences(fraction);

    /** Stored like this:
     * [outsForSeq1Timestep1][outsForSeq2Timestep1] .. [outsForSeqNTimestep1][outsForSeq1Timestep2] ..
     */
    m_precedingLayerOutputMatrix = helpers::Matrix<TDevice>(&this->precedingLayer().outputs(), this->precedingLayer().size(), this->curMaxSeqLength() * this->parallelSequences());

    forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
    for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
        forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

        int rows = this->size() / (m_isBidirectional ? 2 : 1);
        int cols = this->curMaxSeqLength() * this->parallelSequences();

        fwbw->igActsMatrix           = helpers::Matrix<TDevice>(&fwbw->igActs           , rows, cols);
        fwbw->hiddenTmpOutputsMatrix = helpers::Matrix<TDevice>(&fwbw->hiddenTmpOutputs , rows, cols);

        fwbw->igDeltasMatrix         = helpers::Matrix<TDevice>(&fwbw->igDeltas         , rows, cols);
        fwbw->hiddenTmpDeltasMatrix  = helpers::Matrix<TDevice>(&fwbw->hiddenTmpErrors  , rows, cols);
    }
}

template <typename TDevice>
void RNNLayer<TDevice>::computeForwardPass()
{

    // sum up the activations from the preceding layer
    // forward states
    m_fw.igActsMatrix.assignProduct(m_fw.weightMatrices.igInput, true, m_precedingLayerOutputMatrix, false);

    // compute the block outputs
    int els = this->size() / (m_isBidirectional ? 2 : 1);
    int n   = this->parallelSequences() * els;

    // compute internal states
    internal::ComputeBlockOutputFn fn;
    fn.igActs = helpers::getRawPointer(m_fw.igActs);
    fn.bias   = this->bias();
    fn.biasWeights = this->_rawBiasWeights;
    fn.els    = els;

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

    // backward states
    if (m_isBidirectional) {
        m_bw.igActsMatrix.assignProduct(m_bw.weightMatrices.igInput, true, m_precedingLayerOutputMatrix, false);


        // compute the block outputs
        int els = this->size() / (m_isBidirectional ? 2 : 1);
        int n   = this->parallelSequences() * els;

        // compute internal states
        fn.igActs = helpers::getRawPointer(m_bw.igActs);
        fn.biasWeights += els;

        for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
            // collect outputs from previous timestep
            if (timestep != this->curMaxSeqLength()-1) {
                m_bw.timestepMatrices[timestep].igActs.addProduct(m_bw.weightMatrices.igInternal, true, m_bw.timestepMatrices[timestep+1].hiddenTmpOutputs, false);
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
    }

    // resort outputs
    if (m_isBidirectional) {
        internal::ResortOutputsFn fn;
        fn.layerSize    = this->size();
        fn.effLayerSize = this->size() / 2;
        fn.fwOutputs    = helpers::getRawPointer(m_fw.hiddenTmpOutputs);
        fn.bwOutputs    = helpers::getRawPointer(m_bw.hiddenTmpOutputs);

        thrust::transform(
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(0) + this->curMaxSeqLength() * this->parallelSequences() * this->size(),
                    this->_outputs().begin(),
                    fn
                    );
    }
    else {
        this->_outputs().swap(m_fw.hiddenTmpOutputs);
    }
}

template <typename TDevice>
void RNNLayer<TDevice>::computeBackwardPass() {

    if (m_isBidirectional) {
        internal::ResortOutputErrorsFn fn;
        fn.layerSize      = this->size();
        fn.effLayerSize   = this->size() / 2;
        fn.fwOutputErrors = helpers::getRawPointer(m_fw.hiddenTmpErrors);
        fn.bwOutputErrors = helpers::getRawPointer(m_bw.hiddenTmpErrors);

        int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin(),   thrust::counting_iterator<int>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin()+n, thrust::counting_iterator<int>(0)+n)),
            fn
            );
    }
    else {
        m_fw.hiddenTmpErrors.swap(this->outputs());
        m_fw.hiddenTmpErrors.swap(this->outputErrors());
    }

    // calculate the block errors
    {{
        int els = this->size() / (m_isBidirectional ? 2 : 1);
        int n   = this->parallelSequences() * els;

        // forward states
        internal::ComputeBlockErrorsFn fn;
        fn.effLayerSize       = els;
        fn.prevOutputDistance = -n;
        fn.patTypes           = helpers::getRawPointer(this->patTypes());
        fn.igActs             = helpers::getRawPointer(m_fw.igActs);
        fn.igDeltas           = helpers::getRawPointer(m_fw.igDeltas);

        for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
            // collect errors from previous timestep
            if (timestep != this->curMaxSeqLength()-1) {
                m_fw.timestepMatrices[timestep].hiddenTmpErrors.addProduct(m_fw.weightMatrices.igInternal, false, m_fw.timestepMatrices[timestep+1].igDeltas, false);
            }

            // compute errors
            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(m_fw.hiddenTmpErrors.begin() + n*timestep,   thrust::counting_iterator<int>(n*timestep),   thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1),   thrust::constant_iterator<bool>(!timestep),   thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                thrust::make_zip_iterator(thrust::make_tuple(m_fw.hiddenTmpErrors.begin() + n*timestep+n, thrust::counting_iterator<int>(n*timestep)+n, thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1)+n, thrust::constant_iterator<bool>(!timestep)+n, thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)),
                fn
                );
        }

        // backward states
        if (m_isBidirectional) {
            fn.prevOutputDistance = +n;
            fn.igActs             = helpers::getRawPointer(m_bw.igActs);
            fn.igDeltas           = helpers::getRawPointer(m_bw.igDeltas);

            for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
                // collect errors from previous timestep
                if (timestep != 0) {
                    m_bw.timestepMatrices[timestep].hiddenTmpErrors.addProduct(m_bw.weightMatrices.igInternal, false, m_bw.timestepMatrices[timestep-1].igDeltas, false);
                }

                // compute errors
                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(m_bw.hiddenTmpErrors.begin() + n*timestep,   thrust::counting_iterator<int>(n*timestep),   thrust::constant_iterator<bool>(!timestep),   thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1),   thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                    thrust::make_zip_iterator(thrust::make_tuple(m_bw.hiddenTmpErrors.begin() + n*timestep+n, thrust::counting_iterator<int>(n*timestep)+n, thrust::constant_iterator<bool>(!timestep)+n, thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1)+n, thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)),
                    fn
                    );
            }
        }
    }}

    // back-propagate the error to the preceding layer
    {{
        TrainableLayer<TDevice> *pl = dynamic_cast<TrainableLayer<TDevice>*>(&this->precedingLayer());
        if (pl) {
            helpers::Matrix<TDevice> plErrorsMatrix(&pl->outputErrors(), pl->size(), this->curMaxSeqLength() * this->parallelSequences());

            // forward states
            plErrorsMatrix.addProduct   (m_fw.weightMatrices.igInput, false, m_fw.igDeltasMatrix, false);

            // backward states
            if (m_isBidirectional) {
                plErrorsMatrix.addProduct(m_bw.weightMatrices.igInput, false, m_bw.igDeltasMatrix, false);
            }
        }
    }}

    // compute the weight updates
    {{
        internal::ComputeWeightUpdateFn fn;
        fn.layerSize             = this->size();
        fn.effLayerSize          = this->size() / (m_isBidirectional ? 2 : 1);
        fn.precLayerSize         = this->precedingLayer().size();
        fn.timestepDistance      = this->parallelSequences() * this->size() / (m_isBidirectional ? 2 : 1);
        fn.parallelSequences     = this->parallelSequences();
        fn.patternsCount         = this->curMaxSeqLength() * this->parallelSequences();
        fn.biasWeightsOffset     = this->size() * this->precedingLayer().size();
        fn.internalWeightsOffset = fn.biasWeightsOffset + this->size();
        fn.bias                  = this->bias();
        fn.plOutputs             = helpers::getRawPointer(this->precedingLayer().outputs());
        fn.fwOutputs             = helpers::getRawPointer(m_fw.hiddenTmpOutputs);
        fn.bwOutputs             = helpers::getRawPointer(m_bw.hiddenTmpOutputs);
        fn.fwIgDeltas            = helpers::getRawPointer(m_fw.igDeltas);
        fn.bwIgDeltas            = helpers::getRawPointer(m_bw.igDeltas);

        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(0) + (int)this->weightUpdates().size(),
            this->_weightUpdates().begin(),
            fn
            );
    }}

    // re-swap the output errors and the tmp output errors of the forward pass
    if (!m_isBidirectional) {
        this->outputErrors().swap(m_fw.hiddenTmpErrors);
        this->_outputs()    .swap(m_fw.hiddenTmpOutputs);
    }

}

// explicit template instantiations
template class RNNLayer<Cpu>;
template class RNNLayer<Gpu>;

}
