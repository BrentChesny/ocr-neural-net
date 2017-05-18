#include "MNIST.h"
#include "Network.h"

int main() {
    // Load the MNIST dataset
    std::shared_ptr<MNIST_Dataset> dataset = MNIST::load_dataset();

    // Create our neural network
    Network network;
    network.add_layer(784, 30);
    network.add_layer(30, 10);

    // Create our trainer
    StochasticGradientDescentTrainer trainer(dataset, 30, 10, 3.0f);

    // Train the network
    network.train(trainer);

    return 0;
}