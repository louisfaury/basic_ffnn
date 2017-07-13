/* file : main.cpp
 * author : Louis Faury
 * date : 08/06/17
 */

#include "neuron/Neuron.h"
#include "layer/SigmoidLayer.h"
#include "neural_network/NeuralNetwork.h"
#include "iostream"
#include "dataset/Dataset.h"
#include <string>
#include <random>
#include <ctime>

// TODO : trainer class
// TODO : add bias !

int main(int argc, char** argv)
{
    srand(time(0));
    /*
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
    */

    std::string sinDataFile = "../src/dataset/sin_dataset.txt";
    Dataset ds;
    if ( ds.load(sinDataFile) )
    {
        std::cout << "Success loading data" << "\n";
        int sampleSize = 5;
        Eigen::MatrixXd in(sampleSize,ds.getInputSize());
        Eigen::MatrixXd out(sampleSize,ds.getOutputSize());
        ds.sample(sampleSize, in, out);
        std::cout << in <<'\n' << '\n';
        ds.sample(sampleSize,in,out);
        std::cout << in <<'\n';
    }
}
