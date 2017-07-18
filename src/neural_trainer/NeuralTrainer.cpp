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

