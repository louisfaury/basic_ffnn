/* file : Layer.h
 * author : Louis Faury
 * date : 08/06/17
 */

#include "vector"
#include "Neuron.h"
#include "Eigen/Dense"

#ifndef LAYER_H
#define LAYER_H

// alias
using NeuralArray = std::vector<Neuron*>;
using NeuralArrayIt = NeuralArray::iterator;

class Layer
{
public:
    Layer();
    Layer(int size);
    ~Layer();

    virtual Eigen::VectorXd    getOutputs();
    virtual void        setActivations(VectorXd aa);
    virtual int         getSize(){return m_size;}
    virtual void        mock() = 0;
protected:
    int m_size;
    NeuralArray m_neurons;

};

#endif // LAYER_H
