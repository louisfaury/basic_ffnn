/* file : SigmoidNeuron.h
 * author : Louis Faury
 * date : 08/06/17
 */

#include "Neuron.h"

#ifndef SIGMOIDNEURON_H
#define SIGMOIDNEURON_H

class SigmoidNeuron : public Neuron
{
public:
    SigmoidNeuron();

protected:
    virtual void _computeOutput();
};

#endif // SIGMOIDNEURON_H
