/* file : Layer.cpp
 * author : Louis Faury
 * date : 08/06/17
 */

#include "Layer.h"
#include "assert.h"

using namespace Eigen;

Layer::Layer() : m_size(0)
{
}

Layer::Layer(int size) : m_size(size)
{
}

Layer::~Layer()
{
    for (NeuralArrayIt it = m_neurons.begin(); it != m_neurons.end(); it++)
    {
        delete(*it);
    }
}

VectorXd Layer::getOutputs()
{
    VectorXd res(m_size);
    int i(0);
    for (NeuralArrayIt it = m_neurons.begin(); it != m_neurons.end(); it++)
    {
        res(i) = (*it)->getOutput();
        ++i;
    }

    return res;
}

VectorXd Layer::getDerivativeActivations()
{
    VectorXd res(m_size);
    int i(0);
    for (NeuralArrayIt it = m_neurons.begin(); it != m_neurons.end(); it++)
    {
        res(i) = (*it)->getDerivativeActivation();
        ++i;
    }

    return res;
}

void Layer::setActivations(VectorXd aa)
{
    int size = aa.rows();
    if (size!=m_size)
        printf("Wrong dimensions (Layer::setActivations)\n");
    else
    {
        NeuralArrayIt it = m_neurons.begin();
        for (int i=0; i<size; i++)
        {
            (*it)->setActivation(aa(i));
            it++;
        }
    }
}

