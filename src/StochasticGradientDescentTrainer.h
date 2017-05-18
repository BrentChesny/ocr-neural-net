//
// Created by Brent Chesny on 18/05/2017.
//

#ifndef OCR_NEURAL_NET_STOCHASTICGRADIENTDESCENTTRAINER_H
#define OCR_NEURAL_NET_STOCHASTICGRADIENTDESCENTTRAINER_H


#include "MNIST.h"
#include "Trainer.h"
#include "CommandLineReporter.h"

class StochasticGradientDescentTrainer : public Trainer
{
public:
    StochasticGradientDescentTrainer(std::shared_ptr<MNIST_Dataset> dataset,
                                     int epochs,
                                     int batch_size,
                                     float learning_rate,
                                     std::unique_ptr<Reporter> reporter = std::unique_ptr<Reporter>(new CommandLineReporter))
            : _dataset(dataset), _epochs(epochs), _batch_size(batch_size),
              _learning_rate(learning_rate), _reporter(std::move(reporter)) {}

    void train(Network &network) const;

private:
    void train_mini_batch(Network &network, unsigned int begin_index, unsigned int end_index) const;
    std::pair<std::vector<Matrix>, std::vector<Matrix>> backpropagate(Network &network, const Matrix &x, const Matrix &y) const;
    Matrix cost_derivative(const Matrix &output, const Matrix &expected_result) const;
    float evaluate(Network &network) const;

    std::shared_ptr<MNIST_Dataset> _dataset;
    int _epochs;
    int _batch_size;
    float _learning_rate;
    std::unique_ptr<Reporter> _reporter;
};


#endif //OCR_NEURAL_NET_STOCHASTICGRADIENTDESCENTTRAINER_H
