/* file : LinearNeuron.h
 * author : Louis Faury
 * date : 08/06/17
 */

#include "Neuron.h"

#ifndef LINEARNEURON_H
#define LINEARNEURON_H


class LinearNeuron : public Neuron
{
public:
    LinearNeuron();

protected:
    virtual void _computeOutput();
};

#endif // LINEARNEURON_H
