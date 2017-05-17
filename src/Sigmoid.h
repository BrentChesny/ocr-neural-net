//
// Created by Brent Chesny on 17/05/2017.
//

#ifndef OCR_NEURAL_NET_SIGMOID_H
#define OCR_NEURAL_NET_SIGMOID_H

#include "ActivatorFunction.h"

class Sigmoid : public ActivatorFunction
{
public:
    virtual float apply(float z);
    virtual float apply_derivative(float z);
};


#endif //OCR_NEURAL_NET_SIGMOID_H
