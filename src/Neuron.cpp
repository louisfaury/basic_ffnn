/* file : Neuron.cpp
 * author : Louis Faury
 * date : 08/06/17
 */

#include "Neuron.h"

Neuron::Neuron() : m_a(0.), m_o(0.)
{ 
}

double Neuron::getOutput()
{
    _computeOutput();
    return m_o;
}

Neuron::~Neuron()
{
}

