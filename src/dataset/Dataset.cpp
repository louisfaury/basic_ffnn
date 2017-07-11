/* file : Dataset.cpp
 * author : Louis Faury
 * date : 09/07/17
 */

#include "Dataset.h"
#include <fstream>
#include <sstream>

Dataset::Dataset() : m_inputSize(0), m_outputSize(0)
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

