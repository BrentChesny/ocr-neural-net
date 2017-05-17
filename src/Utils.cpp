//
// Created by Brent Chesny on 17/05/2017.
//

#include "Utils.h"

#include <cstdlib>
#include <iostream>

float Utils::random()
{
    return -1.0 + 2.0f * ((float) rand() / (RAND_MAX));
}

Vec Utils::random_vector(size_t length)
{
    Vec vec;

    for (unsigned int j = 0; j < length; ++j) {
        vec.push_back(Utils::random());
    }

    return vec;
}

Mat Utils::random_matrix(size_t rows, size_t cols)
{
    Mat mat;

    for (unsigned int i = 0; i < rows; ++i) {
        mat.push_back(Utils::random_vector(cols));
    }

    return mat;
}


Mat Utils::transpose(const Mat &mat)
{
    Mat result;

    for (unsigned int i = 0; i < mat[0].size(); ++i) {
        Vec row;

        for (unsigned int j = 0; j < mat.size(); ++j) {
            row.push_back(mat[j][i]);
        }

        result.push_back(row);
    }

    return result;
}

float Utils::dot(const Vec &vec1, const Vec &vec2)
{
    if (vec1.size() != vec2.size()) {
        std::cerr << "Attempting to calculate dot product of different sized vectors!" << std::endl;
        return 0.0f;
    }

    float result = 0.0f;

    for (unsigned int i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }

    return result;
}

Vec Utils::mat_mul(const Mat &mat, const Vec &vec)
{
    Vec result;

    for (unsigned int i = 0; i < mat.size(); ++i) {
        result.push_back(Utils::dot(mat[i], vec));
    }

    return result;
}


Mat Utils::mat_add(const Mat &mat1, const Mat &mat2)
{
    Mat result;

    for (unsigned int i = 0; i < mat1.size(); ++i) {
        result.push_back(Utils::vec_add(mat1[i], mat2[i]));
    }

    return result;
}

Mat Utils::mat_diff(const Mat &mat1, const Mat &mat2)
{
    Mat result;

    for (unsigned int i = 0; i < mat1.size(); ++i) {
        result.push_back(Utils::vec_diff(mat1[i], mat2[i]));
    }

    return result;
}

Mat Utils::mat_mul_scalar(const Mat &mat, float scalar)
{
    Mat result;

    for (unsigned int i = 0; i < mat.size(); ++i) {
        result.push_back(Utils::vec_mul_scalar(mat[i], scalar));
    }

    return result;
}

Vec Utils::zero_vector(size_t length)
{
    Vec vec;

    for (unsigned int j = 0; j < length; ++j) {
        vec.push_back(0.0f);
    }

    return vec;
}

Vec Utils::zero_vector(const Vec &vec)
{
    return Utils::zero_vector(vec.size());
}

Mat Utils::zero_matrix(size_t rows, size_t cols)
{
    Mat mat;

    for (unsigned int i = 0; i < rows; ++i) {
        mat.push_back(Utils::zero_vector(cols));
    }

    return mat;
}

Mat Utils::zero_matrix(const Mat &mat)
{
    return Utils::zero_matrix(mat.size(), mat[0].size());
}

Vec Utils::vec_add(const Vec &vec1, const Vec &vec2)
{
    if (vec1.size() != vec2.size()) {
        std::cerr << "Attempting to add different sized vectors!" << std::endl;
        return Vec();
    }

    Vec result;

    for (unsigned int i = 0; i < vec1.size(); ++i) {
        result.push_back(vec1[i] + vec2[i]);
    }

    return result;
}

Vec Utils::vec_diff(const Vec &vec1, const Vec &vec2)
{
    if (vec1.size() != vec2.size()) {
        std::cerr << "Attempting to subtract different sized vectors!" << std::endl;
        return Vec();
    }

    Vec result;

    for (unsigned int i = 0; i < vec1.size(); ++i) {
        result.push_back(vec1[i] - vec2[i]);
    }

    return result;
}

Vec Utils::vec_mul_elements(const Vec &vec1, const Vec &vec2)
{
    if (vec1.size() != vec2.size()) {
        std::cerr << "Attempting to multiply different sized vectors!" << std::endl;
        return Vec();
    }

    Vec result;

    for (unsigned int i = 0; i < vec1.size(); ++i) {
        result.push_back(vec1[i] * vec2[i]);
    }

    return result;
}

Mat Utils::vec_mul_t(const Vec &vec1, const Vec &vec2)
{
    Mat result;

    for (unsigned int i = 0; i < vec1.size(); ++i) {
        result.push_back(Utils::vec_mul_scalar(vec2, vec1[i]));
    }

    return result;
}

Vec Utils::vec_mul_scalar(const Vec &vec, float scalar)
{
    Vec result;

    for (unsigned int i = 0; i < vec.size(); ++i) {
        result.push_back(vec[i] * scalar);
    }

    return result;
}

void Utils::print(const Vec &vec)
{
    std::cerr << "[ ";

    for (unsigned int i = 0; i < vec.size(); ++i) {
        std::cerr << vec[i] << " ";
    }

    std::cerr << "]" << std::endl;
}