//
// Created by Brent Chesny on 19/05/2017.
//

#ifndef OCR_NEURAL_NET_COMMANDLINEREPORTER_H
#define OCR_NEURAL_NET_COMMANDLINEREPORTER_H

#include <iostream>
#include "Reporter.h"

class CommandLineReporter : public Reporter
{
public:
    virtual void progress(int epoch, float progress, std::ostream &out = std::cout);
    virtual void result(int epoch, float accuracy, std::ostream &out = std::cout);
private:
    void print_progress_bar(float progress, std::ostream &out = std::cout);
};


#endif //OCR_NEURAL_NET_COMMANDLINEREPORTER_H
