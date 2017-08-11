/* file : NeuralTrainer.cpp
 * author : Louis Faury
 * date : 13/07/17
 * brief : Simple dataset class for data storage (NUMERICAL ATTRIBUTES ONLY !)
 */

#include "NeuralTrainer.h"
#include "../neural_network/NeuralNetwork.h"
#include "Eigen/Core"

#define EPS 0.00001

using namespace Opt_na;
using namespace Eigen;

NeuralTrainer::NeuralTrainer()
{
}

void NeuralTrainer::addData(std::__cxx11::string dataFileName, double ttRatio)
{
    Dataset ds;
    ds.load(dataFileName);
    ds.split(ttRatio,m_trainSet,m_testSet);
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
    if (mbSize==0 || mbSize >= m_trainSet.getSampleSize())
    {
        m_mbSize = m_trainSet.getSampleSize();
    }
    else
        m_mbSize = mbSize;
}

void NeuralTrainer::train(NeuralNetwork *net)
{
    // options
    int cost    = m_opt.cost;
    int opti    = m_opt.opt;
    int inS     = m_trainSet.getInputSize();
    int outS    = m_trainSet.getOutputSize();
    bool gradCheck = false;

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
    double trainLoss,testLoss;

    // training
    for (int i=0; i<m_maxIter; i++)
    {
        m_trainSet.sample(m_mbSize,X,y); // mini-batch sampling
        /* Computing gradient */
        dW *= 0;
        for (int m=0; m<m_mbSize; m++)
        {
            X1 = (X.block(m,0,inS,1) ).transpose();
            y1 = (y.block(m,0,outS,1) ).transpose();
            switch (cost)
            {
            case (int)Cost_en::SSE:
                diff = net->feedForward(X1) - y1;
                break;
            default:
                /// TODO
                break;
            }
            dw1 = net->backPropagate(diff);
           dW += dw1;
        }
        if (gradCheck)
        {
            MatrixXd thDw= _finiteDiffGrad(X,y,net);
            MatrixXd dDw = thDw - dW;
            double err = dDw.squaredNorm();
            if (err > EPS)
            {
                printf("Wrong grad computation. Norm error : %f\n",err);
            }
            else
            {
                printf("Gradient computation ok. Norm error : %f\n",err);
            }
        }

        /* Update */
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

        trainLoss = evaluateTrainLoss(net);
        testLoss = evaluateTestLoss(net);
        printf("Epoch %i/%i, \t loss (training) %4.8f, \t loss (testing) %4.4f\n",i,m_maxIter,trainLoss,testLoss);
    }
}

double NeuralTrainer::evaluateTrainLoss(NeuralNetwork *net)
{
    return _evaluateLoss(net,m_trainSet);
}

double NeuralTrainer::evaluateTestLoss(NeuralNetwork *net)
{
    return _evaluateLoss(net,m_testSet);
}

double NeuralTrainer::_evaluateLoss(NeuralNetwork *net, SubDataset ds)
{
    double res(0.);

    // test data querry
    int size = ds.getSampleSize();
    Eigen::MatrixXd in, out;
    ds.batch(in,out);

    switch (m_opt.cost)
    {
    case (int)Cost_en::SSE:
    {
        Eigen::MatrixXd pred, ref, inSample,diff;
        for (int i=0; i<size; i++)
        {
            inSample = in.block(i,0,1,ds.getInputSize());
            pred = net->feedForward(inSample);
            ref = out.block(i,0,1,ds.getOutputSize());
            diff = ref - pred;
            res += 0.5 * diff.squaredNorm();
        }
        break;
    }
    default:
        /// TODO
        break;
    }

    return res;
}

VectorXd NeuralTrainer::_finiteDiffGrad(MatrixXd X, MatrixXd y, NeuralNetwork *net)
{
    int size = net->getVectorSize();
    int inS  = m_trainSet.getInputSize();
    int outS = m_trainSet.getOutputSize();

    Eigen::VectorXd dW(size), dW1(size);
    Eigen::VectorXd w = net->net2Vec();
    VectorXd X1(inS);
    VectorXd y1(outS);
    dW.setZero();

    for (int m=0; m<m_mbSize; m++)
    {
        X1 = (X.block(m,0,inS,1) ).transpose();
        y1 = (y.block(m,0,outS,1) ).transpose();

        // assuming SSE loss for now /// TODO implement others
        Eigen::VectorXd dW1(size), diff, wDw(size), e(size);
        double err, errdw;
        for (int i=0; i<size; i++)
        {
            net->vec2Net(w);
            e.setZero();
            e(i,0) = EPS;
            diff = net->feedForward(X1)-y1;
            err = 0.5*diff.squaredNorm();
            wDw = w + e;
            net->vec2Net(wDw);
            diff = net->feedForward(X1)-y1;
            errdw = 0.5*diff.squaredNorm();
            dW1(i,0) = (errdw - err)/EPS;
        }
        dW += dW1;
    }

    return dW;
}

