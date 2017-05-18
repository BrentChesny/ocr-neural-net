//
// Created by Brent Chesny on 16/05/2017.
//

#include "MNIST.h"

#include <fstream>

const std::string MNIST::MNIST_TRAIN_IMAGE_FILE = "../data/train-images-idx3-ubyte";
const std::string MNIST::MNIST_TRAIN_LABEL_FILE = "../data/train-labels-idx1-ubyte";
const std::string MNIST::MNIST_TEST_IMAGE_FILE = "../data/t10k-images-idx3-ubyte";
const std::string MNIST::MNIST_TEST_LABEL_FILE = "../data/t10k-labels-idx1-ubyte";

MNIST_Dataset MNIST::load_dataset()
{
    MNIST_Dataset dataset;

    load_image_file(MNIST_TRAIN_IMAGE_FILE, dataset.train_images);
    std::cout << "Loaded training images: " << dataset.train_images.size() << std::endl;

    load_label_file(MNIST_TRAIN_LABEL_FILE, dataset.train_labels);
    std::cout << "Loaded training labels: " << dataset.train_labels.size() << std::endl;

    load_image_file(MNIST_TEST_IMAGE_FILE, dataset.test_images);
    std::cout << "Loaded test images: " << dataset.test_images.size() << std::endl;

    load_label_file(MNIST_TEST_LABEL_FILE, dataset.test_labels);
    std::cout << "Loaded test labels: " << dataset.test_labels.size() << std::endl;

    return dataset;
}

void MNIST::load_image_file(const std::string &path, std::vector<Matrix> &images)
{
    auto buffer = read_file(path, 0x803);

    auto count = read_header(buffer, 1);
    auto rows = read_header(buffer, 2);
    auto columns = read_header(buffer, 3);

    auto pixel_buffer = reinterpret_cast<uint8_t *>(buffer.get() + 16);

    for (unsigned int i = 0; i < count; ++i) {
        std::vector<float> image;

        for (unsigned int j = 0; j < rows * columns; ++j) {
            uint8_t value = *(pixel_buffer + i * rows * columns + j);
            float normalized_value = (float) value / 255.0f;

            image.push_back(normalized_value);
        }

        images.push_back(Matrix::create(image, rows*columns));
    }
}

void MNIST::load_label_file(const std::string &path, std::vector<Matrix> &labels)
{
    auto buffer = read_file(path, 0x801);

    auto count = read_header(buffer, 1);

    auto label_buffer = reinterpret_cast<uint8_t *>(buffer.get() + 8);

    for (unsigned int i = 0; i < count; ++i) {
        uint8_t value = *(label_buffer + i);
        labels.push_back(one_hot_encode(value));
    }
}

std::unique_ptr<uint8_t[]> MNIST::read_file(const std::string& path, uint32_t key)
{
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
        return {};
    }

    auto size = file.tellg();
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[size]);

    // Read the entire file at once
    file.seekg(0, std::ios::beg);
    file.read((char *) buffer.get(), size);
    file.close();

    auto magic = read_header(buffer, 0);

    if (magic != key) {
        std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        return {};
    }

    return buffer;
}

uint32_t MNIST::read_header(const std::unique_ptr<uint8_t[]>& buffer, size_t position)
{
    auto header = reinterpret_cast<uint32_t*>(buffer.get());

    auto value = *(header + position);
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}

Matrix MNIST::one_hot_encode(uint8_t value)
{
    std::vector<float> vec;

    for (uint8_t i = 0; i < 10; ++i) {
        float bit = (value == i) ? 1.0f : 0.0f;

        vec.push_back(bit);
    }

    return Matrix::create(vec, vec.size());
}
