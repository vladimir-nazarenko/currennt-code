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
 * @param TDevice The computation device (Cpu or Gpu)
 *********************************************************************************************/
    template <typename TDevice>
    class RNNLayer: public TrainableLayer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;


        struct weight_matrices_t {
            helpers::Matrix<TDevice> igInput;
            helpers::Matrix<TDevice> igInternal;
        };

        struct timestep_matrices_t {
            helpers::Matrix<TDevice> igActs;
            helpers::Matrix<TDevice> igDeltas;
            helpers::Matrix<TDevice> tmpOutputs;
            helpers::Matrix<TDevice> tmpErrors;
        };

        struct forward_backward_info_t {
            real_vector tmpOutputs;
            real_vector tmpErrors;
            real_vector igActs;
            real_vector igDeltas;

            helpers::Matrix<TDevice> igActsMatrix;
            helpers::Matrix<TDevice> igDeltasMatrix;
            helpers::Matrix<TDevice> tmpOutputsMatrix;
            helpers::Matrix<TDevice> tmpErrorsMatrix;

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

        int gradientThreshold() const;
        void loadSequences(const data_sets::DataSetFraction &fraction);
        void computeForwardPass();
        void computeBackwardPass();

        virtual ~RNNLayer();

    private:
        forward_backward_info_t m_fw;
        forward_backward_info_t m_bw;
        real_t m_gradThreshold;
        helpers::Matrix<TDevice> m_precedingLayerOutputMatrix;
        bool m_isBidirectional;
        real_t *_rawBiasWeights;
    };

}

#endif // RNNLAYER_H
