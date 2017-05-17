//
// Created by Brent Chesny on 17/05/2017.
//

#ifndef OCR_NEURAL_NET_NETWORK_H
#define OCR_NEURAL_NET_NETWORK_H


#include "Layer.h"
#include "MNIST.h"

#include <vector>

class Network
{
public:
    Network();

    Vec feedforward(const Vec &input);
    void train(const MNIST_Dataset &dataset, int epochs, int batch_size, float learning_rate);

private:
    size_t _num_layers;
    std::vector<Vec> _biases;
    std::vector<Mat> _weights;

    void train_mini_batch(const MNIST_Dataset &dataset, size_t begin_index, size_t end_index, float learning_rate);
    std::pair<std::vector<Vec>, std::vector<Mat>> backpropagate(const Vec &x, const Vec &y);
    float evaluate(const MNIST_Dataset &dataset);

    float sigmoid(float x);
    Vec sigmoid(const Vec &x);
    Vec sigmoid_prime(const Vec &x);
    Vec cost_derivative(const Vec &output, const Vec& expected_result);
};


#endif //OCR_NEURAL_NET_NETWORK_H
