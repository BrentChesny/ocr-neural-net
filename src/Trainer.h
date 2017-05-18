//
// Created by Brent Chesny on 18/05/2017.
//

#ifndef OCR_NEURAL_NET_TRAINER_H
#define OCR_NEURAL_NET_TRAINER_H

class Network;

class Trainer {
public:
    virtual void train(Network &network) const = 0;
};



#endif //OCR_NEURAL_NET_TRAINER_H
