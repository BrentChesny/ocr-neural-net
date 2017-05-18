//
// Created by Brent Chesny on 17/05/2017.
//

#include "Network.h"

Network::Network() : _num_layers(0)
{
}

void Network::add_layer(size_t input_size, size_t layer_size, ActivatorFunction* activator)
{
    _biases.push_back(Matrix::random(layer_size));
    _weights.push_back(Matrix::random(layer_size, input_size));
    _activators.push_back(activator);

    _num_layers += 1;
}

Matrix Network::feedforward(const Matrix &input)
{
    Matrix output = input;

    for (unsigned int i  = 0; i < _num_layers; ++i) {
        output = _weights[i] * output + _biases[i];
        output = _activators[i]->apply(output);
    }

    return output;
}

void Network::train(const Trainer &trainer)
{
    trainer.train(*this);
}
