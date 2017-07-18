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

    bool load(std::string fileName);
    void sample(int sampleSize, Eigen::MatrixXd &inSamples, Eigen::MatrixXd &outSamples); // no replacement
    void batch(Eigen::MatrixXd &inSamples, Eigen::MatrixXd &outSamples);

    int getInputSize(){ return m_inputSize; }
    int getOutputSize(){ return m_outputSize; }
    int getSampleSize(){ return m_numSamples; }

protected:
    Eigen::MatrixXd m_inputs;   // (x1,..,xN)^T
    Eigen::MatrixXd m_outputs;  // (y1,..,yN)^T
    int m_inputSize;
    int m_outputSize;
    int m_numSamples;
};

#endif // DATASET_H
