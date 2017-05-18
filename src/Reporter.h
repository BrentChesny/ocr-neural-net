//
// Created by Brent Chesny on 19/05/2017.
//

#ifndef OCR_NEURAL_NET_REPORTER_H
#define OCR_NEURAL_NET_REPORTER_H

class Reporter
{
public:
    virtual void progress(int epoch, float progress, std::ostream &out = std::cout) = 0;
    virtual void result(int epoch, float accuracy, std::ostream &out = std::cout) = 0;
};

#endif //OCR_NEURAL_NET_REPORTER_H
