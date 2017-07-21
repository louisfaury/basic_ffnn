/* file : NeuralTrainer.h
 * author : Louis Faury
 * date : 13/07/17
 * brief : Simple dataset class for data storage (NUMERICAL ATTRIBUTES ONLY !)
 */

#include "../dataset/Dataset.h"
#include <string>

#ifndef NEURALTRAINER_H
#define NEURALTRAINER_H

class NeuralNetwork; // feed-forward declaration

namespace Opt_na
{
    enum class Cost_en
    {
        SSE=0, // Sum of Squared Error (linear with Gaussian noise)
        CE   // Cross-entropy, binary classification
    };
    enum class Optimization_en
    {
        classic =0, // Nothing fancy
        momentum,
        adam
    };
}


struct Options
{   // option index
    int cost;
    int opt;
};


class NeuralTrainer
{
public:
    NeuralTrainer();

    void addData(std::string dataFileName);
    void setCostFonction(int costFctIdx);
    void setOptimizationTools(int optiOptIdx);
    void setMiniBatchSize(uint mbSize);
    void setMaxIter(uint maxIter){ m_maxIter = maxIter; }
    void setLearningRate(double lr){ m_lr = lr; }

    void train(NeuralNetwork* net); // TODO
protected:
    Dataset m_ds;
    Options m_opt;
    int m_maxIter;
    int m_mbSize;
    int m_lr;

};

#endif // NEURALTRAINER_H
