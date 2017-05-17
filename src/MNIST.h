//
// Created by Brent Chesny on 16/05/2017.
//

#ifndef OCR_NEURAL_NET_MNIST_H
#define OCR_NEURAL_NET_MNIST_H

#include <string>
#include <vector>

typedef std::vector<float> Image;

struct MNIST_Dataset
{
    std::vector<Image> train_images;
    std::vector<std::vector<float>> train_labels;
    std::vector<Image> test_images;
    std::vector<std::vector<float>> test_labels;
};

class MNIST
{
public:
    static const std::string MNIST_TRAIN_IMAGE_FILE;
    static const std::string MNIST_TRAIN_LABEL_FILE;
    static const std::string MNIST_TEST_IMAGE_FILE;
    static const std::string MNIST_TEST_LABEL_FILE;

    static MNIST_Dataset load_dataset();

private:
    static void load_image_file(const std::string &path, std::vector<Image> &images);
    static void load_label_file(const std::string &path, std::vector<std::vector<float>> &labels);
    static std::vector<float> one_hot_encode(uint8_t value);

    static std::unique_ptr<uint8_t[]> read_file(const std::string& path, uint32_t key);
    static uint32_t read_header(const std::unique_ptr<uint8_t[]> &buffer, size_t position);
};


#endif //OCR_NEURAL_NET_MNIST_H
