//
// Created by Brent Chesny on 18/05/2017.
//

#include <cstdlib>
#include <iomanip>

#include "Matrix.h"

Matrix::Matrix(size_t rows, size_t cols, float val) : _rows(rows), _cols(cols)
{
    _data.resize(rows * cols, val);
}

Matrix::Matrix(size_t rows, size_t cols, std::vector<float> data) : _rows(rows), _cols(cols), _data(data)
{
}

Matrix Matrix::zeroes(size_t rows, size_t cols)
{
    return Matrix(rows, cols);
}

Matrix Matrix::random(size_t rows, size_t cols)
{
    Matrix mat(rows, cols);

    for (unsigned int i = 0; i < mat._data.size(); ++i) {
        mat._data[i] = -1.0f + 2.0f * ((float) rand() / RAND_MAX);
    }

    return mat;
}

Matrix Matrix::create(std::vector<float> data, size_t rows, size_t cols)
{
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Size of data does not match given matrix dimensions.");
    }

    return Matrix(rows, cols, data);
}

void Matrix::print(std::ostream &out) const
{
    out << "[";
    for (unsigned int row = 0; row < _rows; ++row) {
        if (row != 0) out << " ";

        out << "[";
        for (unsigned int col = 0; col < _cols; ++col) {
            out << std::fixed << std::setprecision(5) << std::setw(8) << _data[row * _cols + col] << " ";
        }
        out << "]";

        if (row != _rows - 1) out << std::endl;
    }
    out << "]" << std::endl;
}

Matrix Matrix::transpose() const
{
    Matrix transposed(_cols, _rows);

    for (unsigned int row = 0; row < _rows; ++row) {
        for (unsigned int col = 0; col < _cols; ++col) {
            transposed._data[col * _rows + row] = _data[row * _cols + col];
        }
    }

    return transposed;
}

Matrix Matrix::map(std::function<float(float)> func) const
{
    Matrix result(_rows, _cols);

    for (unsigned int row = 0; row < _rows; ++row) {
        for (unsigned int col = 0; col < _cols; ++col) {
            result._data[row * _cols + col] = func(_data[row * _cols + col]);
        }
    }

    return result;
}

Matrix Matrix::operator*(const Matrix &other) const
{
    if (_cols != other._rows) {
        throw std::invalid_argument("Attempting to multiply matrices with non-matching dimensions.");
    }

    Matrix result(_rows, other._cols);

    for (unsigned int row = 0; row < _rows; ++row) {
        for (unsigned int col = 0; col < other._cols; ++col) {
            for (unsigned int k = 0; k < _cols; ++k) {
                result._data[row * other._cols + col] += _data[row * _cols + k] * other._data[k * other._cols + col];
            }
        }
    }

    return result;
}

Matrix Matrix::operator+(const Matrix &other) const
{
    if (_rows != other._rows || _cols != other._cols) {
        throw std::invalid_argument("Attempting to add matrices with non-matching dimensions.");
    }

    Matrix result(_rows, _cols);

    for (unsigned int row = 0; row < _rows; ++row) {
        for (unsigned int col = 0; col < _cols; ++col) {
            result._data[row * _cols + col] = _data[row * _cols + col] + other._data[row * _cols + col];
        }
    }

    return result;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    if (_rows != other._rows || _cols != other._cols) {
        throw std::invalid_argument("Attempting to subtract matrices with non-matching dimensions.");
    }

    Matrix result(_rows, _cols);

    for (unsigned int row = 0; row < _rows; ++row) {
        for (unsigned int col = 0; col < _cols; ++col) {
            result._data[row * _cols + col] = _data[row * _cols + col] - other._data[row * _cols + col];
        }
    }

    return result;
}

Matrix Matrix::operator*(float scalar) const
{
    Matrix result(_rows, _cols);

    for (unsigned int row = 0; row < _rows; ++row) {
        for (unsigned int col = 0; col < _cols; ++col) {
            result._data[row * _cols + col] = _data[row * _cols + col] * scalar;
        }
    }

    return result;
}

Matrix Matrix::hadamard(const Matrix &other) const
{
    if (_rows != other._rows || _cols != other._cols) {
        throw std::invalid_argument("Attempting to compute Hadamard product of matrices with non-matching dimensions.");
    }

    Matrix result(_rows, _cols);

    for (unsigned int row = 0; row < _rows; ++row) {
        for (unsigned int col = 0; col < _cols; ++col) {
            result._data[row * _cols + col] = _data[row * _cols + col] * other._data[row * _cols + col];
        }
    }

    return result;
}

unsigned int Matrix::max_arg()
{
    if (_cols != 1) {
        throw std::invalid_argument("Attempting to compute max-arg of non-vector.");
    }

    unsigned int max_index = 0;
    for (unsigned int j = 1; j < _data.size(); ++j) {
        if (_data[j] > _data[max_index])
            max_index = j;
    }

    return max_index;
}

float Matrix::get(unsigned int row, unsigned int col)
{
    if (row * _cols + col >= _data.size()) {
        throw std::out_of_range("Trying to access non-existent element.");
    }

    return _data[row * _cols + col];
}


