#include "NeuralNetwork.h"
#include <random>

using namespace Eigen;

NeuralNetwork::NeuralNetwork() : m_depth(0)
{
}

NeuralNetwork::NeuralNetwork(int inSize, int outSize) : m_depth(0), m_inSize(inSize), m_outSize(outSize)
{
}

NeuralNetwork::~NeuralNetwork()
{
    for (LayerArrayIt it = m_layers.begin(); it != m_layers.end(); it++)
        delete(*it);
}

void NeuralNetwork::addHiddenLayer(Layer *layer)
{
    m_layers.push_back(layer);

    // creating weight matrix
    int rows(0), cols(0);
    if (m_depth==0)
    {
        cols = m_inSize;
        rows = layer->getSize();
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
    Layer* prevLayer = m_layers.at(m_depth-1);
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
                val = 2*range*(((float)std::rand()/RAND_MAX)-0.5);
                (*it)(i,j) = val;
            }
        }
    }
}

VectorXd NeuralNetwork::feedForward(VectorXd in)
{
    VectorXd out(m_outSize);
    if (in.rows() != m_inSize)
    {
        printf("Wrong input size : %i vs %i\n",(int)in.rows(),m_inSize);
    }
    else
    {
        WeightsIt wit = m_weights.begin();
        LayerArrayIt lit = m_layers.begin();

        // feed-forward
        VectorXd hid = (*wit)*in;
        for (int d=0; d<m_depth; d++)
        {
            wit++;
            (*lit)->setActivations(hid);
            hid = (*wit)*((*lit)->getOutputs());
        }

        out = hid;
        return out;
    }
}

VectorXd NeuralNetwork::backPropagate(VectorXd diff)
{
    /// TODO
}

int NeuralNetwork::getVectorSize()
{
    int res(0);
    for (WeightsIt it = m_weights.begin(); it != m_weights.end(); it++)
    {
        res += (it->rows()) * (it->cols());
    }
    return res;
}

VectorXd NeuralNetwork::net2Vec()
{
    int size = getVectorSize();
    VectorXd res(size);

    int idx(0);
    MatrixXd tmp;
    for (WeightsIt wit = m_weights.begin(); wit != m_weights.end(); wit++)
    {
        tmp = *wit;
        tmp.transposeInPlace(); // we fill the vector row by row
        for (int j=0; j<tmp.cols(); j++)
        {
            res.block(idx,0,tmp.rows(),1) = tmp.block(0,j,tmp.rows(),1);
            idx += tmp.rows();
        }
    }

    return res;
}

void NeuralNetwork::vec2Net(VectorXd w)
{
    // little check
    int size = getVectorSize();
    if (w.rows() != size)
    {
        printf("'vec2Net' wrong parameters size : %i vs %i",size,(int)w.rows());
        assert(false);
    }
    // else we update the net
    else
    {
        MatrixXd nW = w.transpose();
        int idx(0);
        for (WeightsIt wit = m_weights.begin(); wit != m_weights.end(); wit++)
        {
            for (int i=0; i<wit->rows(); i++)
            {
                wit->block(i,0,1,wit->cols()) = nW.block(0,idx,1,wit->cols());
                idx += wit->cols();
            }
        }
    }
}
