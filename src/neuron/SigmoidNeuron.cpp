/* file : SigmoidNeuron.cpp
 * author : Louis Faury
 * date : 08/06/17
 */

#include "SigmoidNeuron.h"
#include "math.h"

SigmoidNeuron::SigmoidNeuron() : Neuron()
{
}

double SigmoidNeuron::getDerivativeActivation()
{
    double res;
    double sigmoid = 1./(1+exp(-m_a));

    res = sigmoid * (1-sigmoid);
    return res;
}

void SigmoidNeuron::_computeOutput()
{
    // sigmoidal activation function
    m_o = 1./(1+exp(-m_a));
}



