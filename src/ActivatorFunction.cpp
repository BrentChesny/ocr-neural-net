//
// Created by Brent Chesny on 17/05/2017.
//

#include "ActivatorFunction.h"

Matrix ActivatorFunction::apply(const Matrix &vec)
{
    return vec.map([&](float x) -> float {
        return this->apply(x);
    });
}

Matrix ActivatorFunction::apply_derivative(const Matrix &vec)
{
    return vec.map([&](float x) -> float {
        return this->apply_derivative(x);
    });
}
