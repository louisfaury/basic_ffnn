/* file : SubDataset.h
 * author : Louis Faury
 * date : 10/08/17
 * brief : Simple dataset class for data storage (NUMERICAL ATTRIBUTES ONLY !)
 */

#include "Eigen/Core"
#include "Dataset.h"

#ifndef SUBDATASET_H
#define SUBDATASET_H

class SubDataset : public Dataset
{
public:
    SubDataset();
    void copyData(int numSamples, int inSize, int outSize, Eigen::MatrixXd inSamples, Eigen::MatrixXd outSamples);

};

#endif // SUBDATASET_H
