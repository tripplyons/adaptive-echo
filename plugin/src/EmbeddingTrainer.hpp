#pragma once

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/grad.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <random>
#include <functional>
#include <iostream>

#include "LossGradient.hpp"

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
        for(int epoch = 0; epoch < num_epochs; epoch++){
            double curr_loss = train_step(tau);

            // print progress occasionally
            if(epoch % (num_epochs/10) == 0 || epoch == num_epochs-1){
                std::cout << "  Epoch " << epoch << "/" << num_epochs 
                          << ", Loss: " << current_loss << std::endl;
            }
        }
    }

    // getters that will return the learned embeddings as double vectors
    std::vector<std::vector<double>> get_sound_embeddings() const {
        return get_double_values(sounds);
    }

    std::vector<std::vector<double>> get_setting_embeddings() const {
        return get_double_values(settings);
    }


private:
    int N;
    int embedding_dim;
    double learning_rate;

    std::vector<std::vector<autodiff::var>> sounds;
    std::vector<std::vector<autodiff::var>> settings;

    double train_step(const autodiff::var& tau){
        // compute loss with forward pass from lossgradient.cpp
        autodiff::var loss = forward(tau, sounds, settings);

        // do a backward pass to get a gradient for each variable
        // all_params will store the gradient for each variable of the sounds AND settings matrices
        std::vector<std::reference_wrapper<autodiff::var>> all_params;
        all_params.reserve(N * embedding_dim * 2);

        // fill all_params with current sounds and settings values
        for (auto& row : sounds) {
            for (auto& var : row) {
                all_params.push_back(var);
            }
        }
        for (auto& row : settings) {
            for (auto& var : row) {
                all_params.push_back(var);
            }
        }

        // gradients stores the gradient for each variable in all_params
        auto gradients = autodiff::grad(loss, all_params);

        // descend the gradient and update parameters
        for(size_t i = 0; i < all_params.size(); i++){
            // take step in opposite direction of loss gradient to decrease loss
            all_params[i].get() -= learning_rate * gradients[i];
        }

        return autodiff::val(loss);
    }

    std::vector<std::vector<double>> get_double_values(
        const std::vector<std::vector<autodiff::var>>& matrix) const
    {
        std::vector<std::vector<double>> double_matrix(
            N, std::vector<double>(embedding_dim));
        
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < embedding_dim; ++j) {
                double_matrix[i][j] = autodiff::val(matrix[i][j]);
            }
        }
        return double_matrix;
    }

};