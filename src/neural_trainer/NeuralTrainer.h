/* file : NeuralTrainer.h
 * author : Louis Faury
 * date : 13/07/17
 * brief : Simple dataset class for data storage (NUMERICAL ATTRIBUTES ONLY !)
 */

#include "../dataset/Dataset.h"
#include <string>

#ifndef NEURALTRAINER_H
#define NEURALTRAINER_H

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

    void addData(std::string dataFileName); // TODO
    void setCostFonction(int costFctIdx); // TODO
    void setOptimizationTools(int optiOptIdx); // TODO

protected:
    Dataset ds;
    Options opt;
};

#endif // NEURALTRAINER_H
