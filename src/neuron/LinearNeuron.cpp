#include "LinearNeuron.h"

LinearNeuron::LinearNeuron() : Neuron()
{

}

void LinearNeuron::_computeOutput()
{
    // identity activation
    m_o = m_a;
}

