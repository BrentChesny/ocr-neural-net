//
// Created by Brent Chesny on 17/05/2017.
//

#include "ActivatorFunction.h"

Vec ActivatorFunction::apply(const Vec &vec)
{
    Vec result;

    for (unsigned int i = 0; i < vec.size(); ++i) {
        result.push_back(this->apply(vec[i]));
    }

    return result;
}

Vec ActivatorFunction::apply_derivative(const Vec &vec)
{
    Vec result;

    for (unsigned int i = 0; i < vec.size(); ++i) {
        result.push_back(this->apply_derivative(vec[i]));
    }

    return result;
}
