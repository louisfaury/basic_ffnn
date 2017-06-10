/* file : NeuralNetwork.h
 * author : Louis Faury
 * date : 08/06/17
 */

#include <vector>
#include "Layer.h"
#include "Eigen/Dense"

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

// alias
using LayerArray = std::vector<Layer*>;
using Weights = std::vector<Eigen::MatrixXd>;

class NeuralNetwork
{
public:
    NeuralNetwork();
    //NeuralNetwork(int depth, string cmd);

    // TODO : add predict

protected:
    int m_depth;            // number of layers
    LayerArray m_layers;
    Weights m_weights;      // weights of the net

};

#endif // NEURALNETWORK_H
