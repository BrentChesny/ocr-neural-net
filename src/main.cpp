#include <iostream>

#include "MNIST.h"

int main() {
    MNIST_Dataset dataset = MNIST::load_dataset();

    return 0;
}