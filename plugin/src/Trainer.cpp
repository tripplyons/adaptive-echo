#define _USE_MATH_DEFINES
#include <cmath>

#include "Normalization.hpp"
#include "WavHandler.hpp"

#include <iostream>
#include <sstream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

int main() {
    try {
        std::string model_path = "../adaptive_echo_python/graphs/synth.pt";

        torch::jit::Module module;
        try {
            module = torch::jit::load(model_path);
        } catch (const c10::Error &e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            std::cerr << "Make sure you've run export_graphs.py to generate "
                         "the .pt file"
                      << std::endl;
            return 1;
        }

        module.eval();

        for (const auto &pair : module.named_parameters()) {
            const std::string &name = pair.name;
            torch::Tensor param = pair.value;

            auto random_param = torch::randn_like(param).detach();
            random_param.set_requires_grad(false);

            std::istringstream name_stream(name);
            std::string segment;
            std::vector<std::string> path_parts;

            while (std::getline(name_stream, segment, '.')) {
                path_parts.push_back(segment);
            }

            torch::jit::Module current_module = module;
            for (size_t i = 0; i < path_parts.size() - 1; ++i) {
                current_module = current_module.attr(path_parts[i]).toModule();
            }

            current_module.setattr(path_parts.back(), random_param);
        }

        int64_t sample_rate = 48000;
        double duration = 5.0;
        int64_t num_samples = static_cast<int64_t>(sample_rate * duration);

        auto time_values = torch::linspace(0.0, duration, num_samples);
        auto input = time_values.unsqueeze(0);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        auto output = module.forward(inputs);

        if (output.isTensor()) {
            auto output_tensor = output.toTensor();

            auto has_nan = torch::isnan(output_tensor).any().item<bool>();
            auto has_inf = torch::isinf(output_tensor).any().item<bool>();
            if (has_nan) {
                std::cerr << "Warning: Output contains NaN values!"
                          << std::endl;
            }
            if (has_inf) {
                std::cerr << "Warning: Output contains Inf values!"
                          << std::endl;
            }

            auto output_cpu = output_tensor.cpu().contiguous();

            torch::Tensor output_1d;
            if (output_cpu.dim() == 2 && output_cpu.size(0) == 1) {
                output_1d = output_cpu[0];
            } else if (output_cpu.dim() == 1) {
                output_1d = output_cpu;
            } else {
                output_1d = output_cpu.flatten();
            }

            auto output_double = output_1d.to(torch::kFloat64);
            int64_t actual_num_samples = output_double.size(0);

            std::vector<double> audio_data;
            audio_data.reserve(actual_num_samples);

            auto output_ptr = output_double.data_ptr<double>();
            for (int64_t i = 0; i < actual_num_samples; ++i) {
                audio_data.push_back(output_ptr[i]);
            }

            std::vector<int32_t> normalized_data = normalize(audio_data);

            std::string output_filename = "output.wav";
            writeData(output_filename, normalized_data, sample_rate);
            std::cout << "Successfully wrote WAV file to " << output_filename << "!" << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
