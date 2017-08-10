/* file : Neuron.h
 * author : Louis Faury
 * date : 08/06/17
 */

#ifndef NEURON_H
#define NEURON_H


class Neuron
{
public:
    Neuron();
    virtual double getOutput();
    virtual double getDerivativeActivation() = 0;
    virtual double setActivation(double a){m_a = a;}
    ~Neuron();

protected:
    virtual void _computeOutput() = 0;

    double m_o;
    double m_a;
};

#endif // NEURON_H
