//
// Created by Brent Chesny on 17/05/2017.
//

#ifndef OCR_NEURAL_NET_ACTIVATORFUNCTION_H
#define OCR_NEURAL_NET_ACTIVATORFUNCTION_H

#include "Matrix.h"

class ActivatorFunction
{
public:
    virtual float apply(float z) = 0;
    virtual float apply_derivative(float z) = 0;
    Matrix apply(const Matrix& vec);
    Matrix apply_derivative(const Matrix& vec);
};


#endif //OCR_NEURAL_NET_ACTIVATORFUNCTION_H
