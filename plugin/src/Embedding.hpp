#pragma once

#include <vector>
#include <autodiff/reverse/var.hpp>
#include <stdexcept>

class Embedding{
    public:
        std::vector<std::vector<autodiff::var>>
        MatrixMult(const std::vector<std::vector<autodiff::var>> &matrix,
                   const std::vector<std::vector<autodiff::var>> &input) const{
            if (matrix[0].size() != input.size()) {
                throw std::runtime_error("Matrices are not compatible");
            }
            const int N = matrix[0].size();
            const std::pair<int,int> sz = std::make_pair(matrix.size(),input[0].size());
            std::vector<std::vector<autodiff::var>>
            output(matrix.size(),std::vector<autodiff::var>(input[0].size()));
            for (int i = 0;i<sz.first;i++) {
                for (int j = 0;j<sz.second;j++){
                    autodiff::var sum = 0;
                    for (int k = 0; k < N; k++) {
                        sum += matrix[i][k] * input[k][j];
                    }
                    output[i][j] = sum;
                }
            }
            return output;
        }
};