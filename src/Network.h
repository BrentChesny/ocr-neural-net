//
// Created by Brent Chesny on 17/05/2017.
//

#ifndef OCR_NEURAL_NET_NETWORK_H
#define OCR_NEURAL_NET_NETWORK_H


#include "MNIST.h"
#include "ActivatorFunction.h"
#include "Sigmoid.h"
#include "StochasticGradientDescentTrainer.h"

#include <vector>

class Network
{
public:
    Network();

    void add_layer(size_t input_size, size_t layer_size, ActivatorFunction* activator = new Sigmoid);
    void train(const Trainer &trainer);
    Matrix feedforward(const Matrix &input);

    size_t num_layers() const { return _num_layers; }
    std::vector<Matrix>& biases() { return _biases; }
    std::vector<Matrix>& weights() { return _weights; }
    std::vector<ActivatorFunction*>& activators() { return _activators; }

private:
    size_t _num_layers;
    std::vector<Matrix> _biases;
    std::vector<Matrix> _weights;
    std::vector<ActivatorFunction*> _activators;
};


#endif //OCR_NEURAL_NET_NETWORK_H
