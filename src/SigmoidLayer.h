/* file : SigmoidLayer.h
 * author : Louis Faury
 * date : 08/06/17
 */

#ifndef SIGMOIDLAYER_H
#define SIGMOIDLAYER_H

#include "Layer.h"

class SigmoidLayer : public Layer
{
public:
    SigmoidLayer();
    SigmoidLayer(int size);

    virtual void mock(){}
};

#endif // SIGMOIDLAYER_H
