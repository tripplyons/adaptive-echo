#include <autodiff/reverse/var.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <random>
#include <functional>
#include <iostream>
#include <chrono>

#include "LossGradient.cpp"

class EmbeddingTrainer {
public:
    EmbeddingTrainer(int num_pairs, int dim, double lr){
        N = num_pairs;
        embedding_dim = dim;
        learning_rate = lr;

        // construct sounds and settings matrices with random values

        std::mt19937 generator(
            std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        for (int i = 0; i < N; i++) {

            std::vector<autodiff::var> random_vector_1(embedding_dim);
            std::vector<autodiff::var> random_vector_2(embedding_dim);
            for (int j = 0; j < embedding_dim; j++) {
                random_vector_1[j] = distribution(generator);
                random_vector_2[j] = distribution(generator);
            }

            sounds.push_back(random_vector_1);
            settings.push_back(random_vector_2);
        }
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



};