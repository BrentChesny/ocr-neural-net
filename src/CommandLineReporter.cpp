//
// Created by Brent Chesny on 19/05/2017.
//

#include <iomanip>
#include "CommandLineReporter.h"

void CommandLineReporter::progress(int epoch, float progress, std::ostream &out)
{
    out << "Training epoch " << std::setw(2) << epoch << ": ";

    this->print_progress_bar(progress);

    out << " " << int(progress * 100.0) << " %\r";
    out.flush();
}

void CommandLineReporter::result(int epoch, float accuracy, std::ostream &out)
{
    out << "Training epoch " << std::setw(2) << epoch << ": ";

    this->print_progress_bar(1.0f);

    out << " Done - Network accuracy: " << 100 * accuracy << " %" << std::endl;
    out.flush();
}

void CommandLineReporter::print_progress_bar(float progress, std::ostream &out)
{
    int barWidth = 70;

    out << "[";

    int pos = (int) (barWidth * progress);
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) out << "=";
        else if (i == pos) out << ">";
        else out << " ";
    }

    out << "]";
}
