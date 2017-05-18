//
// Created by Brent Chesny on 17/05/2017.
//

#include "Network.h"

#include <iostream>
#include <cmath>

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

void Network::train(const MNIST_Dataset &dataset, int epochs, int batch_size, float learning_rate)
{
    for (int i = 0; i < epochs; ++i) {
        size_t num_samples = dataset.train_images.size();
        size_t num_batches = (num_samples / batch_size);
        num_batches += (num_samples % batch_size == 0) ? 0 : 1;

        for (unsigned int j = 0; j < num_batches; ++j) {
            auto start_index = j * batch_size;
            auto end_index = (j == num_batches - 1) ? num_samples : (j+1) * batch_size;

            this->train_mini_batch(dataset, start_index, end_index, learning_rate);
        }

        std::cout << "Epoch " << i << " complete: " << this->evaluate(dataset) << std::endl;
    }
}

void Network::train_mini_batch(const MNIST_Dataset &dataset, size_t begin_index, size_t end_index, float learning_rate)
{
    // Generate containers for gradient of weights and biases
    std::vector<Matrix> gradient_b;
    for (unsigned int i = 0; i < _num_layers; ++i) {
        gradient_b.push_back(Matrix::zeroes(_biases[i].rows()));
    }

    std::vector<Matrix> gradient_w;
    for (unsigned int i = 0; i < _num_layers; ++i) {
        gradient_w.push_back(Matrix::zeroes(_weights[i].rows(), _weights[i].cols()));
    }

    size_t batch_size = end_index - begin_index;
    for (size_t i = begin_index; i < end_index; ++i) {
        auto x = dataset.train_images[i];
        auto y = dataset.train_labels[i];

        auto delta_gradient = this->backpropagate(x, y);
        std::vector<Matrix> delta_gradient_b = delta_gradient.first;
        std::vector<Matrix> delta_gradient_w = delta_gradient.second;

        for (unsigned int j = 0; j < gradient_b.size(); ++j) {
            gradient_b[j] = gradient_b[j] + delta_gradient_b[j];
        }

        for (unsigned int j = 0; j < gradient_w.size(); ++j) {
            gradient_w[j] = gradient_w[j] + delta_gradient_w[j];
        }
    }


    for (unsigned int i = 0; i < _num_layers; ++i) {
        _biases[i] = _biases[i] - gradient_b[i] * (learning_rate / batch_size);
        _weights[i] = _weights[i] - gradient_w[i] * (learning_rate / batch_size);
    }
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> Network::backpropagate(const Matrix &x, const Matrix &y)
{
    // Generate containers for gradient of weights and biases
    std::vector<Matrix> gradient_b;
    for (unsigned int i = 0; i < _num_layers; ++i) {
        gradient_b.push_back(Matrix::zeroes(_biases[i].rows()));
    }

    std::vector<Matrix> gradient_w;
    for (unsigned int i = 0; i < _num_layers; ++i) {
        gradient_w.push_back(Matrix::zeroes(_weights[i].rows(), _weights[i].cols()));
    }

    // Feedforward
    Matrix activation = x;

    std::vector<Matrix> activations;
    activations.push_back(x);

    std::vector<Matrix> zs;

    for (unsigned int i = 0; i < _num_layers; ++i) {
        Matrix z = _weights[i] * activation + _biases[i];

        zs.push_back(z);
        activation = _activators[i]->apply(z);
        activations.push_back(activation);
    }

    // Backward pass
    Matrix delta = this->cost_derivative(activations.back(), y).hadamard(_activators[_activators.size() - 1]->apply_derivative(zs.back()));
    gradient_b[gradient_b.size() - 1] = delta;
    gradient_w[gradient_w.size() - 1] = delta * activations[activations.size() - 2].transpose();
    for (unsigned int i = 2; i <= _num_layers; ++i) {
        Matrix z = zs[zs.size() - i];
        Matrix sp = _activators[_activators.size() - i]->apply_derivative(z);

        delta = (_weights[_num_layers - i + 1].transpose() * delta).hadamard(sp);
        gradient_b[gradient_b.size() - i] = delta;
        gradient_w[gradient_w.size() - i] = delta * activations[activations.size() - i - 1].transpose();
    }

    return std::make_pair(gradient_b, gradient_w);
}

float Network::evaluate(const MNIST_Dataset &dataset)
{
    int count = 0;

    for (unsigned int i = 0; i < dataset.test_images.size(); ++i) {
        Matrix x = dataset.test_images[i];
        Matrix y = dataset.test_labels[i];

        unsigned int result = this->feedforward(x).max_arg();

        if (y.get(result) > 0.99f)
            count += 1;
    }

    return (float) count / dataset.test_images.size();
}

Matrix Network::cost_derivative(const Matrix &output, const Matrix &expected_result)
{
    return output - expected_result;
}
