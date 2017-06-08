/* file : SigmoidLayer.cpp
 * author : Louis Faury
 * date : 08/06/17
 */

#include "SigmoidLayer.h"
#include "SigmoidNeuron.h"

SigmoidLayer::SigmoidLayer() : Layer()
{
}

SigmoidLayer::SigmoidLayer(int size) : Layer(size)
{
    for (int i=0; i<size; i++)
    {
        Neuron* nr = new SigmoidNeuron;
        m_neurons.push_back(nr);
    }
}

