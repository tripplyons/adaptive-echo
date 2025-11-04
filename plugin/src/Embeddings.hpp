#include <autodiff/reverse/var.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <random>
#include <functional>
#include <iostream>

#include "LossGradient.cpp"

class EmbeddingTrainer {
public:
    EmbeddingTrainer(int num_pairs, int dim, double lr){
        N = num_pairs;
        embedding_dim = dim;
        learning_rate = lr;

        // construct sounds and settings matrices with random values
    }

    void train(int num_epochs, const autodiff::var& tau){
        // train sounds and settings matrices
    }

    


private:
    int N;
    int embedding_dim;
    double learning_rate;

    std::vector<std::vector<autodiff::var>> sounds;
    std::vector<std::vector<autodiff::var>> settings;



}