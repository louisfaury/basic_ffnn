/* file : NeuralTrainer.cpp
 * author : Louis Faury
 * date : 13/07/17
 * brief : Simple dataset class for data storage (NUMERICAL ATTRIBUTES ONLY !)
 */

#include "NeuralTrainer.h"
#include "../neural_network/NeuralNetwork.h"
#include "Eigen/Core"

using namespace Opt_na;
using namespace Eigen;

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
        m_mbSize = m_ds.getSampleSize();
    }
    else
        m_mbSize = mbSize;
}

void NeuralTrainer::train(NeuralNetwork *net)
{
    // options
    int cost    = m_opt.cost;
    int opti    = m_opt.opt;
    int inS     = m_ds.getInputSize();
    int outS    = m_ds.getOutputSize();

    // csts
    int wSize = net->getVectorSize();   // nn parameter size
    VectorXd w = net->net2Vec();        // parameter initialization
    VectorXd dW(wSize);                 // gradient declaration
    VectorXd dw1(wSize);
    MatrixXd X(m_mbSize,inS);
    MatrixXd y(m_mbSize,outS);
    VectorXd X1(inS);
    VectorXd y1(outS);
    VectorXd diff(outS);
    // training
    for (int i=0; i<m_maxIter; i++)
    {
        m_ds.sample(m_mbSize,X,y); // mini-batch sampling
        dW *= 0;
        for (int m=0; m<m_mbSize; m++)
        {
            X1 = ( X.block(m,0,inS,1) ).transpose();
            y1 = (y.block(m,0,outS,1) ).transpose();
            switch (cost)
            {
            case (int)Cost_en::SSE:
                diff = y1 - net->feedForward(X1);
                break;
            default:
                /// TODO
                break;
            }
            dw1 = net->backPropagate(diff);
            dW += dw1;
        }
        // update the current estimate according to option (constant learning rate, momentum, ..)
        switch (opti)
        {
        case (int)Optimization_en::classic:
            w -= m_lr*dW;
            net->vec2Net(w);
            break;
        default:
            /// TODO
            break;
        }
    }
}

