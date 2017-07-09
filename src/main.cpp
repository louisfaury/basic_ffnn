/* file : main.cpp
 * author : Louis Faury
 * date : 08/06/17
 */

#include "Neuron.h"
#include "SigmoidLayer.h"
#include "LinearLayer.h"
#include "NeuralNetwork.h"
#include "iostream"

// TODO : add bias !

int main(int argc, char** argv)
{
    int inputSize = 1;
    int outputSize = 1;

    NeuralNetwork nn = NeuralNetwork(inputSize,outputSize);
    // defining hidden layers
    SigmoidLayer* in2hid = new SigmoidLayer(10);
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
}
