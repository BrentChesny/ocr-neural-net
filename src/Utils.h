//
// Created by Brent Chesny on 17/05/2017.
//

#ifndef OCR_NEURAL_NET_UTILS_H
#define OCR_NEURAL_NET_UTILS_H

#include <vector>

typedef std::vector<float> Vec;
typedef std::vector<std::vector<float>> Mat;


class Utils
{
public:
    static float random();

    static Vec random_vector(size_t length);
    static Mat random_matrix(size_t rows, size_t cols);

    static Vec zero_vector(size_t length);
    static Vec zero_vector(const Vec& vec);

    static Mat zero_matrix(size_t rows, size_t cols);
    static Mat zero_matrix(const Mat& mat);

    static Mat transpose(const Mat &mat);
    static float dot(const Vec &vec1, const Vec &vec2);
    static Vec mat_mul(const Mat &mat, const Vec &vec);
    static Mat mat_add(const Mat &mat1, const Mat& mat2);
    static Mat mat_diff(const Mat &mat1, const Mat& mat2);
    static Mat mat_mul_scalar(const Mat &mat, float scalar);
    static Vec vec_add(const Vec &vec1, const Vec &vec2);
    static Vec vec_diff(const Vec &vec1, const Vec &vec2);
    static Vec vec_mul_elements(const Vec &vec1, const Vec &vec2);
    static Mat vec_mul_t(const Vec &vec1, const Vec &vec2);
    static Vec vec_mul_scalar(const Vec &vec, float scalar);

    static void print(const Vec &vec);
};


#endif //OCR_NEURAL_NET_UTILS_H
