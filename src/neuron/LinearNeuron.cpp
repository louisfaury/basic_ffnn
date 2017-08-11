#include "LinearNeuron.h"

LinearNeuron::LinearNeuron() : Neuron()
{

}

double LinearNeuron::getDerivativeActivation()
{
    return 1;
}

void LinearNeuron::_computeOutput()
{
    // identity activation
    m_o = m_a;
}

