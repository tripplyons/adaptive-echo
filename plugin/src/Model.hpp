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
    Model(const Layer &input_, const std::vector<Layer> &hidden_layers_, const Layer &output_){
        input = input_;
        hidden_layers = hidden_layers_;
        output = output_;

        num_hidden = hidden_layers.size();

        input_dimension = input.get_dimension();
        output_dimension = output.get_dimension();
        
        for(int i = 0; i < num_hidden; i++){
            hidden_layer_dimension.push_back(hidden_layers[i].get_dimension());
        }
    }




private:
    Layer input;
    std::vector<Layer> hidden_layers;
    Layer output;

    int num_hidden;
    int input_dimension;
    int output_dimension;
    std::vector<int> hidden_layer_dimension; // size num_hidden + 2

}