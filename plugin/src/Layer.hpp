#pragma once

#include <autodiff/reverse/var.hpp>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

class Node {
  public:
    explicit Node(const std::vector<autodiff::var> &weights,
                  const autodiff::var &bias) {
        this->weights = weights;
        this->bias = bias;
    }
    autodiff::var Output(const std::vector<autodiff::var> &input) const {
        if (input.size() != weights.size()) {
            throw std::invalid_argument("Input size mismatch");
        }
        const int N = input.size();
        autodiff::var output = 0.0;
        for (int i = 0; i < N; i++) {
            output += weights[i] * input[i];
        }
        return autodiff::reverse::detail::max(0.0, output + bias);
    }

    std::vector<autodiff::var> weights;
    autodiff::var bias;
};

class Layer {
  public:
    explicit Layer(const int node_count) {

        std::mt19937 generator(
            std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);

        for (int i = 0; i < node_count; i++) {

            std::vector<autodiff::var> random_vector(node_count);
            for (int j = 0; j < node_count; j++) {
                random_vector[j] = distribution(generator);
            }

            nodes.push_back(Node(random_vector, distribution(generator)));
        }
    }
    explicit Layer(const int node_count, const int input_count) {

        std::mt19937 generator(
            std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);

        for (int i = 0; i < node_count; i++) {

            std::vector<autodiff::var> random_vector(node_count);
            for (int j = 0; j < input_count; j++) {
                random_vector[j] = distribution(generator);
            }

            nodes.push_back(Node(random_vector, distribution(generator)));
        }
    }
    explicit Layer(const int node_count,
                   const std::vector<std::vector<autodiff::var>> &weights,
                   const std::vector<autodiff::var> &biases) {

        for (int i = 0; i < node_count; i++) {
            nodes.push_back(Node(weights[i], biases[i]));
        }
    }
    void SetWeights(std::vector<std::vector<autodiff::var>> &weights,
                    std::vector<autodiff::var> biases) {
        int index = 0;
        for (Node node : nodes) {
            node.weights = weights[index];
            node.bias = biases[index];
            index++;
        }
    }
    std::vector<autodiff::var>
    LayerResult(const std::vector<autodiff::var> &inputs) const {
        double rms_total = 0;
        const int N = nodes.size();
        for (autodiff::var input : inputs) {
            rms_total += pow(((double)input), 2);
        }
        autodiff::var rms = sqrt(rms_total / N);
        std::vector<autodiff::var> normal(N);
        for (int i = 0; i < N; i++) {
            normal[i] = inputs[i] / rms;
        }
        std::vector<autodiff::var> result(N);
        for (int i = 0; i < N; i++) {
            result[i] = nodes[i].Output(inputs);
        }
        return result;
    }

  private:
    std::vector<Node> nodes;
};