/* file : Layer.h
 * author : Louis Faury
 * date : 08/06/17
 */

#include "vector"
#include "Neuron.h"

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

protected:
    int m_size;
    NeuralArray m_neurons;
};

#endif // LAYER_H
