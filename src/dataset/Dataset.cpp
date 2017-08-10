/* file : Dataset.cpp
 * author : Louis Faury
 * date : 09/07/17
 */

#include "Dataset.h"
#include <fstream>
#include <sstream>
#include <vector>
#include "SubDataset.h"

Dataset::Dataset() : m_inputSize(0), m_outputSize(0), m_numSamples(0)
{    
}

bool Dataset::load(std::__cxx11::string fileName)
{
    bool res(false);

    std::ifstream file;
    file.open(fileName,std::ifstream::in);

    if (file)
    {
        std::string line, val;
        int in, out, numSamples;

        // in and out respective sizes
        std::getline(file,line);
        std::stringstream sline(line);
        std::getline(sline,val,',');
        in = std::stoi(val);
        std::getline(sline,val,',');
        out = std::stoi(val);
        std::getline(sline,val,',');
        numSamples = std::stoi(val);
        if (in*out*numSamples > 0)
        {
            m_inputSize = in;
            m_outputSize = out;
            m_numSamples = numSamples;
            m_inputs = Eigen::MatrixXd(m_numSamples,m_inputSize);
            m_outputs = Eigen::MatrixXd(m_numSamples,m_outputSize);
        }

        // file reading;
        numSamples *= 0;
        double value(0);
        while ( std::getline(file,line) )
        {
            sline = std::stringstream(line);
            for (int i=0; i<in; i++)
            {
                std::getline(sline,val,',');
                std::string test = sline.str();
                value = std::stod(val);
                m_inputs(numSamples,i) = value;
            }
            for (int j=0; j<out; j++)
            {
                std::getline(sline,val,',');
                value = std::stod(val);
                m_outputs(numSamples,j) = value;
            }
            numSamples++;
        }

        // little safety check, if nothing was triggered by Eigen
        if (numSamples == m_numSamples)
        {
            // sanity flag off
            res = true;
        }
    }
    return res;
}

void Dataset::sample(int sampleSize, Eigen::MatrixXd& inSamples, Eigen::MatrixXd& outSamples)
{

    if (sampleSize > m_numSamples)
    {
        printf("Sample size greater than avalaible samples");
        assert(false);
    }
    else
    {
        if (sampleSize == m_numSamples)
        {   // batch
            inSamples = m_inputs;
            outSamples = m_outputs;
        }
        else
        {
            std::vector<int> fullVector;
            for (int i=0; i<m_numSamples; i++)
                fullVector.push_back(i);
            // not best method but simple :
            std::random_shuffle(fullVector.begin(), fullVector.end() );
            for (int i=0; i<sampleSize; i++)
            {
                inSamples.block(i,0,1,m_inputSize) = m_inputs.block(fullVector.at(i),0,1,m_inputSize);
                outSamples.block(i,0,1,m_outputSize) = m_outputs.block(fullVector.at(i),0,1,m_outputSize);
            }
        }
    }
}

void Dataset::split(double ttRatio, SubDataset &trainSet, SubDataset &testSet)
{
    // sanity check
    assert( (ttRatio>0. && ttRatio<1) );

    // proceed
    int trainSize = ttRatio*m_numSamples;
    int testSize = m_numSamples - trainSize;
    Eigen::MatrixXd inputTrain(trainSize,m_inputSize);
    Eigen::MatrixXd inputTest(testSize,m_inputSize);
    Eigen::MatrixXd outputTrain(trainSize,m_outputSize);
    Eigen::MatrixXd outputTest(testSize,m_outputSize);

    std::vector<int> fullVector;
    for (int i=0; i<m_numSamples; i++)
        fullVector.push_back(i);
    // not best method but simple :
    std::random_shuffle(fullVector.begin(), fullVector.end() );
    for (int i=0; i<m_numSamples; i++)
    {
        if (i<trainSize)
        {
            inputTrain.block(i,0,1,m_inputSize) = m_inputs.block(fullVector.at(i),0,1,m_inputSize);
            outputTrain.block(i,0,1,m_outputSize) = m_outputs.block(fullVector.at(i),0,1,m_outputSize);
        }
        else
        {
            inputTest.block(i-trainSize,0,1,m_inputSize) = m_inputs.block(fullVector.at(i),0,1,m_inputSize);
            outputTest.block(i-trainSize,0,1,m_outputSize) = m_outputs.block(fullVector.at(i),0,1,m_outputSize);
        }
    }

    // copying to subdatasets
    trainSet.copyData(trainSize,m_inputSize,m_outputSize,inputTrain,outputTrain);
    testSet.copyData(testSize,m_inputSize,m_outputSize,inputTest,outputTest);
}

void Dataset::batch(Eigen::MatrixXd &inSamples, Eigen::MatrixXd &outSamples)
{
    sample(m_numSamples,inSamples,outSamples);
}

