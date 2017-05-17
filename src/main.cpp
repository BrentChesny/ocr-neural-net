#include "MNIST.h"
#include "Network.h"


int main() {
    MNIST_Dataset dataset = MNIST::load_dataset();

    Network network;
    network.add_layer(784, 30);
    network.add_layer(30, 10);

    network.train(dataset, 30, 10, 3.0f);

    return 0;
}