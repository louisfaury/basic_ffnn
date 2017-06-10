/* file : LinearLayer.h
 * author : Louis Faury
 * date : 08/06/17
 */

#ifndef LINEARLAYER_H
#define LINEARLAYER_H

#include "Layer.h"

class LinearLayer : public Layer
{
public:
    LinearLayer();
    LinearLayer(int size);

    virtual void mock(){}
};

#endif // LINEARLAYER_H
