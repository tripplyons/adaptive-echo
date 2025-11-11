#pragma once

#include <iostream>
#include <sstream>
#include <torch/optim.h>
#include <torch/script.h>
#include <vector>

using namespace std;

class Synth {
  public:
    Synth(string modelPath) : torchModule(torch::jit::load(modelPath)) {
        torchModule.eval();
    }
    ~Synth() {}
    Synth(const Synth &other) { torchModule = other.torchModule.clone(); }

    void randomizeParameters() {
        for (const auto &pair : torchModule.named_parameters()) {
            auto randomParam = torch::randn_like(pair.value).detach();
            setNestedAttribute(pair.name, randomParam);
        }
    }

    vector<string> getParameterNames() {
        vector<string> names;
        for (const auto &pair : torchModule.named_parameters()) {
            names.push_back(pair.name);
        }
        return names;
    }

    float getParameter(string name) {
        return getNestedAttribute(name).item<float>();
    }

    void setParameter(string name, float value) {
        setNestedAttribute(name, torch::tensor(value));
    }

    torch::Tensor encodeSettings() {
        return torchModule.get_method("encode_settings")({}).toTensor();
    }

    void decodeSettings(torch::Tensor settingsInput) {
        torchModule.get_method("decode_settings")({settingsInput});
    }

    vector<float> generate(vector<float> times) {
        torchModule.eval();
        int64_t numSamples = static_cast<int64_t>(times.size());
        torch::Tensor input =
            torch::from_blob(times.data(), {numSamples},
                             torch::TensorOptions().dtype(torch::kFloat32));
        input = input.unsqueeze(0);

        auto output = torchModule.forward({input});
        if (!output.isTensor()) {
            throw std::runtime_error("Model output is not a tensor");
        }
        auto output_tensor = output.toTensor();
        auto output_cpu = output_tensor.cpu().contiguous().squeeze(0);

        auto output_float = output_cpu.to(torch::kFloat32);

        auto output_ptr = output_float.data_ptr<float>();
        int64_t size = output_float.size(0);
        return vector<float>(output_ptr, output_ptr + size);
    }

    float simpleTrain(vector<float> times, vector<float> &targets,
                      float learningRate = 0.0003f, int numEpochs = 1000,
                      bool log = false) {
        torchModule.train();

        vector<torch::Tensor> parameters;
        for (const auto &pair : torchModule.named_parameters()) {
            parameters.push_back(pair.value);
        }
        torch::optim::OptimizerParamGroup paramGroup(parameters);
        torch::optim::AdamW optimizer(
            parameters, torch::optim::AdamWOptions().lr(learningRate));

        setRequiresGrad(true);

        int64_t numSamples = static_cast<int64_t>(times.size());
        torch::Tensor input =
            torch::from_blob(times.data(), {numSamples},
                             torch::TensorOptions().dtype(torch::kFloat32));
        input = input.unsqueeze(0);
        int64_t numTargets = static_cast<int64_t>(targets.size());
        torch::Tensor target =
            torch::from_blob(targets.data(), {numTargets},
                             torch::TensorOptions().dtype(torch::kFloat32));
        target = target.unsqueeze(0);

        float lossValue = 0;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            auto output = torchModule.forward({input});
            auto output_tensor = output.toTensor();

            auto loss = torch::mse_loss(output_tensor, target);
            lossValue = loss.item<float>();

            if (log) {
                std::cout << "Epoch " << epoch << " loss: " << lossValue
                          << std::endl;
            }

            if (!std::isfinite(lossValue)) {
                std::cerr << "ERROR: Loss is NaN/Inf at epoch " << epoch
                          << std::endl;
                break;
            }

            zeroGrad();

            loss.backward();

            optimizer.step();
        }

        return lossValue;
    }

  private:
    torch::Tensor getNestedAttribute(const std::string &name) {
        std::istringstream name_stream(name);
        std::string segment;
        std::vector<std::string> path_parts;

        while (std::getline(name_stream, segment, '.')) {
            path_parts.push_back(segment);
        }

        torch::jit::Module current_module = torchModule;
        for (size_t i = 0; i < path_parts.size() - 1; ++i) {
            current_module = current_module.attr(path_parts[i]).toModule();
        }

        return current_module.attr(path_parts.back()).toTensor();
    }

    void setNestedAttribute(const std::string &name,
                            const torch::Tensor &value) {
        std::istringstream name_stream(name);
        std::string segment;
        std::vector<std::string> path_parts;

        while (std::getline(name_stream, segment, '.')) {
            path_parts.push_back(segment);
        }

        torch::jit::Module current_module = torchModule;
        for (size_t i = 0; i < path_parts.size() - 1; ++i) {
            current_module = current_module.attr(path_parts[i]).toModule();
        }

        current_module.setattr(path_parts.back(), value);
    }

    void setRequiresGrad(bool requiresGrad) {
        for (const auto &pair : torchModule.named_parameters()) {
            torch::Tensor param = pair.value;
            param.set_requires_grad(requiresGrad);
        }
    }

    void zeroGrad() {
        for (const auto &pair : torchModule.named_parameters()) {
            torch::Tensor param = pair.value;
            if (param.grad().defined()) {
                param.grad().zero_();
            }
        }
    }

    torch::jit::Module torchModule;
};
