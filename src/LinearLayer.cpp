/* file : LinerLayer.cpp
 * author : Louis Faury
 * date : 08/06/17
 */

#include "LinearLayer.h"
#include "LinearNeuron.h"

LinearLayer::LinearLayer() : Layer()
{
}

LinearLayer::LinearLayer(int size) : Layer(size)
{
    for (int i=0; i<size; i++)
    {
        Neuron* nr = new LinearNeuron;
        m_neurons.push_back(nr);
    }
}


