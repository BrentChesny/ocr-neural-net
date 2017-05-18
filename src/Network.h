//
// Created by Brent Chesny on 17/05/2017.
//

#ifndef OCR_NEURAL_NET_NETWORK_H
#define OCR_NEURAL_NET_NETWORK_H


#include "MNIST.h"
#include "ActivatorFunction.h"
#include "Sigmoid.h"

#include <vector>

class Network
{
public:
    Network();

    void add_layer(size_t input_size, size_t layer_size, ActivatorFunction* activator = new Sigmoid);
    void train(const MNIST_Dataset &dataset, int epochs, int batch_size, float learning_rate);
    Matrix feedforward(const Matrix &input);

private:
    size_t _num_layers;
    std::vector<Matrix> _biases;
    std::vector<Matrix> _weights;
    std::vector<ActivatorFunction*> _activators;

    void train_mini_batch(const MNIST_Dataset &dataset, size_t begin_index, size_t end_index, float learning_rate);
    std::pair<std::vector<Matrix>, std::vector<Matrix>> backpropagate(const Matrix &x, const Matrix &y);
    float evaluate(const MNIST_Dataset &dataset);

    Matrix cost_derivative(const Matrix &output, const Matrix &expected_result);
};


#endif //OCR_NEURAL_NET_NETWORK_H
