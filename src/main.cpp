/* file : main.cpp
 * author : Louis Faury
 * date : 08/06/17
 */

#include "neuron/Neuron.h"
#include "layer/SigmoidLayer.h"
#include "neural_network/NeuralNetwork.h"
#include "neural_trainer/NeuralTrainer.h"
#include "iostream"
#include <string>
#include <random>
#include <ctime>

// TODO : add bias !

int main(int argc, char** argv)
{
    srand(time(0));

    /* Testing nn */
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
    nn.randInit(1);

    /* Data */
     std::string sinDataFile = "../src/dataset/sin_dataset.txt";

    /* Neural Training */
    NeuralTrainer neuralTrainer;
    neuralTrainer.addData(sinDataFile);
    neuralTrainer.setCostFonction((int)Opt_na::Cost_en::SSE);
    neuralTrainer.setOptimizationTools((int)Opt_na::Optimization_en::classic);
    neuralTrainer.setMaxIter(10000);
    neuralTrainer.setMiniBatchSize(40);
    neuralTrainer.setLearningRate(0.001);
    neuralTrainer.train(&nn);
}
