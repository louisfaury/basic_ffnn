#include "NeuralNetwork.h"
#include <random>
#include "../layer/LinearLayer.h"
#include <fstream>
#include <stdio.h>

using namespace Eigen;

NeuralNetwork::NeuralNetwork() : m_depth(0)
{
}

NeuralNetwork::NeuralNetwork(int inSize, int outSize) : m_depth(0), m_inSize(inSize), m_outSize(outSize), _SIMLOGFILE("../log/simlog.txt")
{
    LinearLayer* inputLayer = new LinearLayer(m_inSize);
    m_layers.push_back(inputLayer);
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
    Layer* prevLayer = m_layers.at(m_depth);
    int rows = m_outSize;
    int cols = prevLayer->getSize();

    MatrixXd w(rows,cols);
    m_weights.push_back(w);

    LinearLayer* outputLayer = new LinearLayer(m_outSize);
    m_layers.push_back(outputLayer);
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

VectorXd NeuralNetwork::feedForward(VectorXd input)
{
    VectorXd out(m_outSize);
    VectorXd in(m_inSize);
    if (input.cols() != m_inSize)
    {
        printf("Wrong input size : %i vs %i\n",(int)input.cols(),m_inSize);
    }
    else
    {
        WeightsIt wit = m_weights.begin();
        LayerArrayIt lit = m_layers.begin();

        // init
        (*lit)->setActivations(input);
        in = (*lit)->getOutputs();
        lit++;

        // feed-forward
        VectorXd hid = (*wit)*in;
        for (int d=0; d<m_depth; d++)
        {
            wit++;
            (*lit)->setActivations(hid);
            hid = (*wit)*((*lit)->getOutputs());
            lit++;
        }

        (*lit)->setActivations(hid);
        out = (*lit)->getOutputs();
        return out;
    }
}

VectorXd NeuralNetwork::backPropagate(VectorXd diff)
{
    int size(getVectorSize());
    int idx(0);
    VectorXd grad(size);
    VectorXd delta= diff;
    WeightsRIt writ = m_weights.rbegin();
    LayerArrayRIt lrit = m_layers.rbegin();
    lrit++;

    for (int d=0; d<m_depth+1; d++)
    {
        // gradient computation, layer by layer
        VectorXd z = (*lrit)->getOutputs();
        for (int i=0; i<delta.rows(); i++)
        {
            grad.block(size-idx-(i+1)*z.rows(),0,z.rows(),1) = z*delta(delta.rows()-1-i);
        }
        idx += delta.rows()*z.rows();

        // delta backprop
        MatrixXd w = *writ;
        MatrixXd tmp = (w.transpose())*delta;
        delta = VectorXd(z.cols());
        delta = tmp.cwiseProduct((*lrit)->getDerivativeActivations());
        writ++;
        lrit++;
    }

    return grad;
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

void NeuralNetwork::gridSim1d(double lB, double uB, double step)
{/*! Has to be 1D sim ! !*/
    int size = (uB-lB)/step+1;
    double x(lB);
    Eigen::VectorXd v(1),val(1);
    for (int i=0; i<size; i++)
    {
        v.setZero();
        v << x;
        val = feedForward(v);
        _simlog(x,val(0));
        x += step;
    }
}

void NeuralNetwork::_simlog(double x, double y)
{
    std::ofstream f(_SIMLOGFILE,std::ios::app);
    char s[20];
    sprintf(s,"%4.5f,%4.5f\n",x,y);
    f << s;
    f.close();
}
