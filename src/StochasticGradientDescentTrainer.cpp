//
// Created by Brent Chesny on 18/05/2017.
//

#include <iomanip>
#include "StochasticGradientDescentTrainer.h"
#include "Network.h"

void StochasticGradientDescentTrainer::train(Network &network) const
{
    for (int i = 0; i < _epochs; ++i) {
        size_t num_samples = _dataset->train_images.size();
        size_t num_batches = (num_samples / _batch_size);
        num_batches += (num_samples % _batch_size == 0) ? 0 : 1;

        for (unsigned int j = 0; j < num_batches; ++j) {
            unsigned int start_index = j * _batch_size;
            unsigned int end_index = (unsigned int) ((j == num_batches - 1) ? num_samples : (j + 1) * _batch_size);

            this->train_mini_batch(network, start_index, end_index);
            _reporter->progress(i, (float) j / num_batches);
        }

        _reporter->result(i, this->evaluate(network));
    }
}

void StochasticGradientDescentTrainer::train_mini_batch(Network &network, unsigned int begin_index, unsigned int end_index) const
{
    // Generate containers for gradient of weights and biases
    std::vector<Matrix> gradient_b;
    for (unsigned int i = 0; i < network.num_layers(); ++i) {
        gradient_b.push_back(Matrix::zeroes(network.biases()[i].rows()));
    }

    std::vector<Matrix> gradient_w;
    for (unsigned int i = 0; i < network.num_layers(); ++i) {
        gradient_w.push_back(Matrix::zeroes(network.weights()[i].rows(), network.weights()[i].cols()));
    }

    size_t batch_size = end_index - begin_index;
    for (unsigned int i = begin_index; i < end_index; ++i) {
        Matrix x = _dataset->train_images[i];
        Matrix y = _dataset->train_labels[i];

        auto delta_gradient = this->backpropagate(network, x, y);
        std::vector<Matrix> delta_gradient_b = delta_gradient.first;
        std::vector<Matrix> delta_gradient_w = delta_gradient.second;

        for (unsigned int j = 0; j < gradient_b.size(); ++j) {
            gradient_b[j] = gradient_b[j] + delta_gradient_b[j];
        }

        for (unsigned int j = 0; j < gradient_w.size(); ++j) {
            gradient_w[j] = gradient_w[j] + delta_gradient_w[j];
        }
    }


    for (unsigned int i = 0; i < network.num_layers(); ++i) {
        network.biases()[i] = network.biases()[i] - gradient_b[i] * (_learning_rate / batch_size);
        network.weights()[i] = network.weights()[i] - gradient_w[i] * (_learning_rate / batch_size);
    }
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> StochasticGradientDescentTrainer::backpropagate(Network &network, const Matrix &x, const Matrix &y) const
{
    std::vector<Matrix> gradient_b;
    std::vector<Matrix> gradient_w;

    // Feedforward
    Matrix activation = x;
    std::vector<Matrix> activations = {x};
    std::vector<Matrix> zs;

    for (unsigned int i = 0; i < network.num_layers(); ++i) {
        Matrix z = network.weights()[i] * activation + network.biases()[i];
        zs.push_back(z);

        activation = network.activators()[i]->apply(z);
        activations.push_back(activation);
    }

    // Backward pass
    Matrix delta = this->cost_derivative(activations.back(), y).hadamard(network.activators()[network.activators().size() - 1]->apply_derivative(zs.back()));
    gradient_b.insert(gradient_b.begin(), delta);
    gradient_w.insert(gradient_w.begin(), delta * activations[activations.size() - 2].transpose());
    for (unsigned int i = 2; i <= network.num_layers(); ++i) {
        Matrix z = zs[zs.size() - i];
        Matrix sp = network.activators()[network.activators().size() - i]->apply_derivative(z);

        delta = (network.weights()[network.num_layers() - i + 1].transpose() * delta).hadamard(sp);
        gradient_b.insert(gradient_b.begin(), delta);
        gradient_w.insert(gradient_w.begin(), delta * activations[activations.size() - i - 1].transpose());
    }

    return std::make_pair(gradient_b, gradient_w);
}

Matrix StochasticGradientDescentTrainer::cost_derivative(const Matrix &output, const Matrix &expected_result) const
{
    return output - expected_result;
}

float StochasticGradientDescentTrainer::evaluate(Network &network) const
{
    int count = 0;

    for (unsigned int i = 0; i < _dataset->test_images.size(); ++i) {
        Matrix x = _dataset->test_images[i];
        Matrix y = _dataset->test_labels[i];

        unsigned int result = network.feedforward(x).max_arg();

        if (y.get(result) > 0.99f)
            count += 1;
    }

    return (float) count / _dataset->test_images.size();
}