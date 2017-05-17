//
// Created by Brent Chesny on 17/05/2017.
//

#include <cmath>

#include "Sigmoid.h"

float Sigmoid::apply(float z)
{
    return (float) (1.0f / (1.0f + exp(-z)));
}

float Sigmoid::apply_derivative(float z)
{
    return this->apply(z) * (1.0f - this->apply(z));
}
