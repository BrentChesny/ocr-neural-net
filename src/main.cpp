#include <iostream>
#include "MNIST.h"
#include "Network.h"

//class L
//{
//public:
//    L() {
//        _biases.push_back(1.0f);
//        _biases.push_back(2.0f);
//        _biases.push_back(3.0f);
//    }
//
//    void print() {
//        for (int i = 0; i < _biases.size(); ++i) {
//            std::cout << _biases[i] << std::endl;
//        }
//    }
//
//    Vec& biases() { return _biases; };
//private:
//    Vec _biases;
//};

int main() {
    MNIST_Dataset dataset = MNIST::load_dataset();

    Network network;
    network.train(dataset, 30, 10, 3.0f);

//    Vec test;
//    test.push_back(5.0f);
//    test.push_back(6.0f);
//    test.push_back(7.0f);
//
//    L obj;
//    Vec& biases = obj.biases();
//
//    biases = test;
//
//    obj.print();

    return 0;
}