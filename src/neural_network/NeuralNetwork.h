/* file : NeuralNetwork.h
 * author : Louis Faury
 * date : 08/06/17
 */

#include <vector>
#include "../layer/Layer.h"
#include "Eigen/Dense"

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

// alias
using LayerArray = std::vector<Layer*>;
using Weights = std::vector<Eigen::MatrixXd>;
using LayerArrayIt = LayerArray::iterator;
using WeightsIt = Weights::iterator;

class NeuralNetwork
{
public:
    NeuralNetwork();
    NeuralNetwork(int inSize, int outSize);
    // TODO add destructor

    // init fcts
    void addHiddenLayer(Layer* layer);
    void addOutputLayer();
    void zeroInit();
    void randInit(int range);

    // predict fct
    Eigen::VectorXd ffPredict(Eigen::VectorXd in); // feed-forward prediction

protected:
    int m_inSize;           // input size
    int m_outSize;          // output size
    int m_depth;            // number of layers
    LayerArray m_layers;
    Weights m_weights;      // weights of the net

};

#endif // NEURALNETWORK_H
