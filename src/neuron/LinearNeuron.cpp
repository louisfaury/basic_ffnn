#include "LinearNeuron.h"

LinearNeuron::LinearNeuron() : Neuron()
{

}

double LinearNeuron::getDerivativeActivation()
{
    return m_a;
}

void LinearNeuron::_computeOutput()
{
    // identity activation
    m_o = m_a;
}

