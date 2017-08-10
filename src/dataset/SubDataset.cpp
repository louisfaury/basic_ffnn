#include "SubDataset.h"

SubDataset::SubDataset() : Dataset()
{
}

void SubDataset::copyData(int numSamples, int inSize, int outSize, Eigen::MatrixXd inSamples, Eigen::MatrixXd outSamples)
{
    m_numSamples = numSamples;
    m_inputSize = inSize;
    m_outputSize = outSize;
    m_inputs = inSamples;
    m_outputs = outSamples;
}

