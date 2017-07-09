#include "NeuralNetwork.h"
#include <random>

using namespace Eigen;

NeuralNetwork::NeuralNetwork() : m_depth(0)
{
}

NeuralNetwork::NeuralNetwork(int inSize, int outSize) : m_depth(0), m_inSize(inSize), m_outSize(outSize)
{
}

void NeuralNetwork::addHiddenLayer(Layer *layer)
{
    m_layers.push_back(layer);

    // creating weight matrix
    int rows(0), cols(0);
    if (m_depth==0)
    {
        rows = m_inSize;
        cols = layer->getSize();
    }
    else
    {
        Layer* prevLayer = m_layers.at(m_depth);
        rows = layer->getSize();
        cols = prevLayer->getSize();
    }
    MatrixXd w(rows,cols);
    m_weights.push_back(w);

    // incr. depth
    m_depth++;
}

void NeuralNetwork::addOutputLayer()
{
    Layer* prevLayer = m_layers.at(m_depth);
    int rows = m_outSize;
    int cols = prevLayer->getSize();

    MatrixXd w(rows,cols);
    m_weights.push_back(w);
}

void NeuralNetwork::zeroInit()
{
    for (WeightsIt it = m_weights.begin(); it != m_weights.end(); it++)
    {
        it->setZero();
    }
}

void NeuralNetwork::randInit(int range)
{
    double val(0.);
    for (WeightsIt it = m_weights.begin(); it != m_weights.end(); it++)
    {
        for (int i=0; i<it->rows(); i++)
        {
            for (int j=0; j<it->cols(); j++)
            {
                val = 2*range*(std::rand()-0.5);
                (*it)(i,j) = val;
            }
        }
    }
}


