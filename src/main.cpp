/* file : main.cpp
 * author : Louis Faury
 * date : 08/06/17
 */

#include "neuron/Neuron.h"
#include "layer/SigmoidLayer.h"
#include "neural_network/NeuralNetwork.h"
#include "iostream"
#include "dataset/Dataset.h"

// TODO : trainer class
// TODO : add bias !

int main(int argc, char** argv)
{
    int inputSize = 1;
    int outputSize = 1;

    NeuralNetwork nn = NeuralNetwork(inputSize,outputSize);
    // defining hidden layers
    SigmoidLayer* in2hid = new SigmoidLayer(100);
    // adding hidden layers
    nn.addHiddenLayer(in2hid);
    // adding output layer
    nn.addOutputLayer();
    // weight init.
    nn.zeroInit();

    // prediction
    Eigen::Matrix<double,1,1> in;
    in << 1;

    std::cout << "Output : " << nn.ffPredict(in) << '\n';

    Dataset ds();
}
