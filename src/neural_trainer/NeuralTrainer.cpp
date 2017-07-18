/* file : NeuralTrainer.cpp
 * author : Louis Faury
 * date : 13/07/17
 * brief : Simple dataset class for data storage (NUMERICAL ATTRIBUTES ONLY !)
 */

#include "NeuralTrainer.h"

using namespace Opt_na;

NeuralTrainer::NeuralTrainer()
{
}

void NeuralTrainer::addData(std::__cxx11::string dataFileName)
{
    m_ds.load(dataFileName);
}

void NeuralTrainer::setCostFonction(int costFctIdx)
{
    if (costFctIdx != (int)Cost_en::CE && costFctIdx != (int)Cost_en::SSE)
    {
        printf("Wrong cost function index !\n");
        assert(false);
    }
    m_opt.cost = costFctIdx;
}

void NeuralTrainer::setOptimizationTools(int optiOptIdx)
{
    if (optiOptIdx != (int)Optimization_en::adam && optiOptIdx != (int)Optimization_en::momentum && optiOptIdx != (int)Optimization_en::classic)
    {
        printf("Wrong optimization option index !\n");
        assert(false);
    }
    m_opt.opt = optiOptIdx;
}

void NeuralTrainer::setMiniBatchSize(uint mbSize /* 0 if batch, else minibatch size - set to batch if > then the dataset size*/)
{
    if (mbSize==0 || mbSize >= m_ds.getSampleSize())
    {
        m_opt.mbSize = m_ds.getSampleSize();
    }
    else
        m_opt.mbSize = mbSize;
}

