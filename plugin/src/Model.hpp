#pragma once

#include "Layer.hpp"

#include <autodiff/reverse/var.hpp>
#include <chrono>
#include <random>
#include <vector>


// input layer
// list of hidden layers
// out put layers

// number of dimensions for each layer
// number of hidden layers


class Model {
public:
    Model(const std::vector<Layer> &hidden_layers_)
    {
        hidden_layers = hidden_layers_;

        num_hidden = hidden_layers.size();
        
        for(int i = 0; i < num_hidden; i++){
            hidden_layer_dimension.push_back(hidden_layers[i].get_dimension());
        }
    }

    std::vector<autodiff::var> get_output(const std::vector<autodiff::var> &input){

        int input_dimension = input.size();

        if(input_dimension != hidden_layer_dimension[0]){
            throw std::invalid_argument("Input size mismatch (model)");
        }

        std::vector<autodiff::var> current_output = input;

        for(int i = 0; i < num_hidden; i++){
            current_output = hidden_layers[i].LayerResult(current_output);
        }

        return current_output;
    }


private:
    std::vector<Layer> hidden_layers;
    int num_hidden;
    std::vector<int> hidden_layer_dimension;

};