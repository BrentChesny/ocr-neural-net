//
// Created by Brent Chesny on 17/05/2017.
//

#include "Network.h"

#include <iostream>
#include <cmath>

Network::Network()
{
    _num_layers = 2;

    _biases.push_back(Utils::random_vector(30));
    _weights.push_back(Utils::random_matrix(30, 784));

    _biases.push_back(Utils::random_vector(10));
    _weights.push_back(Utils::random_matrix(10, 30));
}

Vec Network::feedforward(const Vec &input)
{
    Vec output = input;
//Utils::print(output);
    for (unsigned int i  = 0; i < _num_layers; ++i) {
  //      Utils::print(_biases[i]);
        output = Utils::mat_mul(_weights[i], output);
        output = Utils::vec_add(output, _biases[i]);
        output = sigmoid(output);

    //    Utils::print(output);
    }
  //  std::cerr << "-------------" << std::endl;
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

        //Utils::print(_biases[0]);

        std::cout << "Epoch " << i << " complete: " << this->evaluate(dataset) << std::endl;
    }
}

void Network::train_mini_batch(const MNIST_Dataset &dataset, size_t begin_index, size_t end_index, float learning_rate)
{
    // Generate containers for gradient of weights and biases
    std::vector<Vec> gradient_b;
    for (unsigned int i = 0; i < _num_layers; ++i) {
        gradient_b.push_back(Utils::zero_vector(_biases[i]));
    }

    std::vector<Mat> gradient_w;
    for (unsigned int i = 0; i < _num_layers; ++i) {
        gradient_w.push_back(Utils::zero_matrix(_weights[i]));
    }

    size_t batch_size = end_index - begin_index;
    //std::cerr << batch_size << "    " << begin_index << "    " << end_index << std::endl;
    for (size_t i = begin_index; i < end_index; ++i) {
        auto x = dataset.train_images[i];
        auto y = dataset.train_labels[i];

        auto delta_gradient = this->backpropagate(x, y);
        std::vector<Vec> delta_gradient_b = delta_gradient.first;
        std::vector<Mat> delta_gradient_w = delta_gradient.second;

        for (unsigned int j = 0; j < gradient_b.size(); ++j) {
            gradient_b[j] = Utils::vec_add(gradient_b[j], delta_gradient_b[j]);
        }

        for (unsigned int j = 0; j < gradient_w.size(); ++j) {
            gradient_w[j] = Utils::mat_add(gradient_w[j], delta_gradient_w[j]);
        }
    }

    //Utils::print(gradient_b[0]);

    for (unsigned int i = 0; i < _num_layers; ++i) {
        //Vec& biases = _layers[i].biases();
        //Mat& weights = _layers[i].weights();
        Vec v = Utils::vec_diff(_biases[i], Utils::vec_mul_scalar(gradient_b[i], learning_rate / batch_size));
        //Utils::print(v);

        _biases[i] = v;
        //Utils::print(_biases[i]);
        _weights[i] = Utils::mat_diff(_weights[i], Utils::mat_mul_scalar(gradient_w[i], learning_rate / batch_size));
    }
    //Utils::print(_layers[0].biases());
}

std::pair<std::vector<Vec>, std::vector<Mat>> Network::backpropagate(const Vec &x, const Vec &y)
{
    // Generate containers for gradient of weights and biases
    std::vector<Vec> gradient_b;
    for (unsigned int i = 0; i < _num_layers; ++i) {
        gradient_b.push_back(Utils::zero_vector(_biases[i]));
    }

    std::vector<Mat> gradient_w;
    for (unsigned int i = 0; i < _num_layers; ++i) {
        gradient_w.push_back(Utils::zero_matrix(_weights[i]));
    }

    // Feedforward
    Vec activation = x;

    std::vector<Vec> activations;
    activations.push_back(x);

    std::vector<Vec> zs;

    for (unsigned int i = 0; i < _num_layers; ++i) {
        Vec b = _biases[i];
        Mat w = _weights[i];

        Vec z = Utils::mat_mul(w, activation);
        z = Utils::vec_add(z, b);

        zs.push_back(z);
        activation = sigmoid(z);
        activations.push_back(activation);
    }

    // Backward pass
    //Utils::print(y);
    Vec spp = sigmoid_prime(zs.back());
    //Utils::print(activations.back());
    //Utils::print(this->cost_derivative(activations.back(), y));
    Vec delta = Utils::vec_mul_elements(this->cost_derivative(activations.back(), y), spp);
    //Utils::print(delta);
    gradient_b[gradient_b.size() - 1] = delta;
    gradient_w[gradient_w.size() - 1] = Utils::vec_mul_t(delta, activations[activations.size() - 2]);
    for (unsigned int i = 2; i <= _num_layers; ++i) {
        Vec z = zs[zs.size() - i];
        Vec sp = sigmoid_prime(z);

        delta = Utils::vec_mul_elements(Utils::mat_mul(Utils::transpose(_weights[_num_layers - i + 1]), delta), sp);
        gradient_b[gradient_b.size() - i] = delta;
        gradient_w[gradient_w.size() - i] = Utils::vec_mul_t(delta, activations[activations.size() - i - 1]);
    }

    return std::make_pair(gradient_b, gradient_w);
}

float Network::evaluate(const MNIST_Dataset &dataset)
{
    int count = 0;

    for (unsigned int i = 0; i < dataset.test_images.size(); ++i) {
        auto x = dataset.test_images[i];
        auto y = dataset.test_labels[i];

        //Utils::print(x);

        Vec result = this->feedforward(x);
        //Utils::print(y);

        int max_index = 0;
        for (unsigned int j = 1; j < result.size(); ++j) {
            if (result[j] > result[max_index])
                max_index = j;
        }

        if (y[max_index] > 0.99f)
            count += 1;
    }

    return (float) count / dataset.test_images.size();
}

float Network::sigmoid(float x)
{
    return (float) (1.0f / (1.0f + exp(-x)));
}

Vec Network::sigmoid(const Vec &vec)
{
    Vec result;

    for (unsigned int i = 0; i < vec.size(); ++i) {
        result.push_back(sigmoid(vec[i]));
    }

    return result;
}

Vec Network::sigmoid_prime(const Vec &vec)
{
    Vec result;

    for (unsigned int i = 0; i < vec.size(); ++i) {
        result.push_back(sigmoid(vec[i]) * (1.0f - sigmoid(vec[i])));
    }

    //Utils::print(result);

    return result;
}

Vec Network::cost_derivative(const Vec &output, const Vec &expected_result)
{
    return Utils::vec_diff(output, expected_result);
}
