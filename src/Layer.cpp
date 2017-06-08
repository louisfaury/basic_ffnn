/* file : Layer.cpp
 * author : Louis Faury
 * date : 08/06/17
 */

#include "Layer.h"

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

