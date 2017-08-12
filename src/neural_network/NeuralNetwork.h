/* file : NeuralNetwork.h
 * author : Louis Faury
 * date : 08/06/17
 */

#include <vector>
#include "../layer/Layer.h"
#include "../neural_trainer/NeuralTrainer.h"
#include "Eigen/Dense"

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

// alias
using LayerArray = std::vector<Layer*>;
using Weights = std::vector<Eigen::MatrixXd>;
using LayerArrayIt = LayerArray::iterator;
using LayerArrayRIt = LayerArray::reverse_iterator;
using WeightsIt = Weights::iterator;
using WeightsRIt = Weights::reverse_iterator;

class NeuralNetwork
{
public:
    NeuralNetwork();
    NeuralNetwork(int inSize, int outSize);
    ~NeuralNetwork();

    // init fcts
    void addHiddenLayer(Layer* layer);
    void addOutputLayer();
    void zeroInit();
    void randInit(int range);

    // predict fct
    Eigen::VectorXd feedForward(Eigen::VectorXd input); // feed-forward prediction

    // backprop
    Eigen::VectorXd backPropagate(Eigen::VectorXd diff);

    // others
    int getVectorSize(); // returns the size of the nn weight parameter
    Eigen::VectorXd net2Vec();
    void vec2Net(Eigen::VectorXd w);
    void gridSim1d(double lB, double uB, double step);

protected:
    void _simlog(double x,double y);

    int m_inSize;           // input size
    int m_outSize;          // output size
    int m_depth;            // number of layers
    LayerArray m_layers;
    Weights m_weights;      // weights of the net

    const std::string _SIMLOGFILE;
};

#endif // NEURALNETWORK_H
