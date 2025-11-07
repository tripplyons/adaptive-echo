#define _USE_MATH_DEFINES
#include <cmath>

#include "WavHandler.hpp"

#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

int main() {
    try {
        // Load the TorchScript model (exported as .pt for C++ compatibility)
        // Path relative to where the Trainer executable is run from
        std::string model_path =
            "../adaptive_echo_python/graphs/audio_encoder.pt";

        std::cout << "Loading model from: " << model_path << std::endl;

        // Load the TorchScript module
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

        module.eval(); // Set to evaluation mode

        std::cout << "Model loaded successfully!" << std::endl;

        // Create a sample input tensor matching the export shape: (1, 48000 *
        // 5) Note: The Encoder model uses nn.Embedding, which expects integer
        // indices Input should be integers in range [0,
        // audio_encoder_input_size)
        int64_t audio_encoder_input_size = 48000 * 5; // 5 seconds at 48kHz
        auto input = torch::randn({1, audio_encoder_input_size});

        std::cout << "Input tensor shape: [" << input.sizes()[0] << ", "
                  << input.sizes()[1] << "]" << std::endl;
        std::cout << "Input tensor dtype: " << input.dtype() << std::endl;

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        auto output = module.forward(inputs);

        std::cout << "Model inference completed!" << std::endl;

        // Print output information if it's a tensor
        if (output.isTensor()) {
            auto output_tensor = output.toTensor();
            std::cout << "Output tensor shape: [";
            for (size_t i = 0; i < output_tensor.sizes().size(); ++i) {
                std::cout << output_tensor.sizes()[i];
                if (i < output_tensor.sizes().size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
            std::cout << "Output tensor dtype: " << output_tensor.dtype()
                      << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
