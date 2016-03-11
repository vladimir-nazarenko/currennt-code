#ifndef RNNLAYER_H
#define RNNLAYER_H

#include "TrainableLayer.hpp"
#include "../helpers/Matrix.hpp"
#include "../helpers/JsonClasses.hpp"

namespace layers {


/******************************************************************************************//**
 * TODO: CHECK ME!
 * weights; with P = precedingLayer().size() and L = size()
 *    ~ weights from preceding layer:
 *        - [0 .. PL-1]:    input weights
 *        - [PL .. PL+LL-1]:  recurrent weights
 *        - [PL + LL .. 2LL+PL-1]: output weigths
 * @param TDevice The computation device (Cpu or Gpu)
 *********************************************************************************************/
    template <typename TDevice>
    class RNNLayer: public TrainableLayer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;


        struct weight_matrices_t {
            helpers::Matrix<TDevice> igInput;
            helpers::Matrix<TDevice> igInternal;
            helpers::Matrix<TDevice> ogOutput;
        };

        struct timestep_matrices_t {
            helpers::Matrix<TDevice> igActs;
            helpers::Matrix<TDevice> igDeltas;
            helpers::Matrix<TDevice> hiddenTmpOutputs;
            helpers::Matrix<TDevice> hiddenTmpErrors;
            helpers::Matrix<TDevice> ogActs;
            helpers::Matrix<TDevice> ogDeltas;
            helpers::Matrix<TDevice> tmpOutputs;
            helpers::Matrix<TDevice> tmpOutputErrors;
        };

        struct forward_backward_info_t {
            real_vector tmpOutputs;
            real_vector tmpOutputErrors;
            real_vector hiddenTmpOutputs;
            real_vector hiddenTmpErrors;
            real_vector igActs;
            real_vector igDeltas;
            real_vector ogActs;
            real_vector ogDeltas;

            helpers::Matrix<TDevice> igActsMatrix;
            helpers::Matrix<TDevice> igDeltasMatrix;
            helpers::Matrix<TDevice> ogActsMatrix;
            helpers::Matrix<TDevice> ogDeltasMatrix;
            helpers::Matrix<TDevice> hiddenTmpOutputsMatrix;
            helpers::Matrix<TDevice> hiddenTmpDeltasMatrix;

            weight_matrices_t weightMatrices;
            weight_matrices_t weightUpdateMatrices;
            std::vector<timestep_matrices_t> timestepMatrices;
        };


        // Layer interface
    public:
        RNNLayer(
                const helpers::JsonValue &layerChild,
                const helpers::JsonValue &weightsSection,
                Layer<TDevice>           &precedingLayer,
                bool bidirectional
                );
        const std::string &type() const;
        /**
         * Sets the values of local variables, inits activation matrices
         */
        void loadSequences(const data_sets::DataSetFraction &fraction);
        void computeForwardPass();
        void computeBackwardPass();

    private:
        forward_backward_info_t m_fw;
        forward_backward_info_t m_bw;
        helpers::Matrix<TDevice> m_precedingLayerOutputMatrix;
        bool m_isBidirectional;
    };

}

#endif // RNNLAYER_H
