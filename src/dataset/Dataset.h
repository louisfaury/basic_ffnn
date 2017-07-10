/* file : Dataset.h
 * author : Louis Faury
 * date : 09/07/17
 * brief : Simple dataset class for data storage (NUMERICAL ATTRIBUTES ONLY !)
 */

#include "Eigen/Core"
#include "string"

#ifndef DATASET_H
#define DATASET_H

class Dataset
{
public:
    Dataset();

    void load(std::string fileName); // TODO
    Eigen::MatrixXd sample(int sampleSize); // TODO
    Eigen::MatrixXd sampleAndReplace(int sampleSize); // TODO


    Eigen::MatrixXd getInputs(){ return m_inputs; }
    Eigen::MatrixXd getOutputs(){ return m_outputs; }
    int getInputSize(){ return m_inputSize; }
    int getOutputSize(){ return m_outputSize; }

protected:
    Eigen::MatrixXd m_inputs;
    Eigen::MatrixXd m_outputs;
    int m_inputSize;
    int m_outputSize;
};

#endif // DATASET_H
