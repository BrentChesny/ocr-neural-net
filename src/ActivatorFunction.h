//
// Created by Brent Chesny on 17/05/2017.
//

#ifndef OCR_NEURAL_NET_ACTIVATORFUNCTION_H
#define OCR_NEURAL_NET_ACTIVATORFUNCTION_H

#include "Utils.h"

class ActivatorFunction
{
public:
    virtual float apply(float z) = 0;
    virtual float apply_derivative(float z) = 0;
    Vec apply(const Vec& vec);
    Vec apply_derivative(const Vec& vec);
};


#endif //OCR_NEURAL_NET_ACTIVATORFUNCTION_H
