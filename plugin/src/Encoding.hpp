#pragma once

#include "Layer.hpp"
#include <vector>


class Network {
    public:
        explicit Network(const int input_nodes, const int output_nodes,
                         const int hidden_nodes, const int hidden_layers) {
            layers.push_back(Layer(hidden_nodes, input_nodes));
            for (int i = 0; i < hidden_layers-2; i++) {
                layers.push_back(Layer(hidden_nodes));
            }
            layers.push_back(Layer(output_nodes, hidden_nodes));
        };
        void SetWeights(std::vector<std::vector<std::vector<autodiff::var>>> &weights,
                        std::vector<std::vector<autodiff::var>> &biases) {
            int index = 0;
            for (Layer layer : layers){
                layer.SetWeights(weights[index], biases[index]);
                index++;
            }
        }
        std::vector<autodiff::var> ForwardPass(const std::vector<autodiff::var> &inputs) {
            std::vector<autodiff::var> next = inputs;
            for (Layer layer : layers) {
                next = layer.LayerResult(next);
            }
            return next;
        }
    private:
        std::vector<Layer> layers;
};