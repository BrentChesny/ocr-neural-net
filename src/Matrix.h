//
// Created by Brent Chesny on 18/05/2017.
//

#ifndef OCR_NEURAL_NET_MATRIX_H
#define OCR_NEURAL_NET_MATRIX_H


#include <vector>
#include <iostream>

class Matrix
{
public:
    static Matrix zeroes(size_t rows, size_t cols = 1);
    static Matrix random(size_t rows, size_t cols = 1);
    static Matrix create(std::vector<float> data, size_t rows, size_t cols = 1);

    size_t rows() const { return _rows; }
    size_t cols() const { return _cols; }
    float get(unsigned int row, unsigned int col = 0);

    void print(std::ostream &out = std::cout) const;

    Matrix transpose() const;
    Matrix map(std::function<float(float)> func) const;
    Matrix operator*(const Matrix &other) const;
    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(float scalar) const;
    Matrix hadamard(const Matrix &other) const;

    unsigned int max_arg();

    // TODO +=, -=, *= operators

private:
    Matrix(size_t rows, size_t cols, float val = 0.0f);
    Matrix(size_t rows, size_t cols, std::vector<float> data);

    size_t _rows;
    size_t _cols;
    std::vector<float> _data;
};


#endif //OCR_NEURAL_NET_MATRIX_H
