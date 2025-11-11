#pragma once

#include <iostream>
#include <sstream>
#include <torch/optim.h>
#include <torch/script.h>
#include <vector>

using namespace std;

class TwoEncoders {
  public:
    TwoEncoders(string modelPath) : torchModule(torch::jit::load(modelPath)) {
        torchModule.eval();
    }
    ~TwoEncoders() {}

    void randomizeParameters() {
        for (const auto &pair : torchModule.named_parameters()) {
            const std::string &name = pair.name;
            torch::Tensor param = pair.value;
            auto randomParam = torch::randn_like(param).detach();
            randomParam.set_requires_grad(false);
            setNestedAttribute(name, randomParam);
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
        return torchModule.attr(name).toTensor().item<float>();
    }

    void setParameter(string name, float value) {
        setNestedAttribute(name, torch::tensor(value));
    }

    torch::Tensor encodeAudio(torch::Tensor audioInput) {
        auto output = torchModule.get_method("encode_audio")({audioInput});
        return output.toTensor().cpu().contiguous().squeeze(0);
    }

    torch::Tensor encodeSettings(torch::Tensor settingsInput) {
        auto output =
            torchModule.get_method("encode_settings")({settingsInput});
        return output.toTensor().cpu().contiguous().squeeze(0);
    }

    torch::Tensor loss(torch::Tensor audioInput, torch::Tensor settingsInput) {
        auto output =
            torchModule.get_method("loss")({audioInput, settingsInput});
        return output.toTensor().cpu().contiguous().squeeze(0);
    }

    float train(int batchSize, float learningRate = 0.0003f,
                      int numSteps = 1000, bool log = false) {
        torchModule.train();

        vector<torch::Tensor> parameters;
        for (const auto &pair : torchModule.named_parameters()) {
            parameters.push_back(pair.value);
        }
        torch::optim::OptimizerParamGroup paramGroup(parameters);
        torch::optim::AdamW optimizer(
            parameters, torch::optim::AdamWOptions().lr(learningRate));

        setRequiresGrad(true);

        int audioInputSize = 48000 * 5;
        int settingsInputSize = 46;

        float lossValue = 0;
        for (int step = 0; step < numSteps; step++) {
            // TODO: these should be loaded from a dataset
            torch::Tensor audioInput = torch::randn({batchSize, audioInputSize}, torch::TensorOptions().dtype(torch::kFloat32));
            torch::Tensor settingsInput = torch::randn({batchSize, settingsInputSize}, torch::TensorOptions().dtype(torch::kFloat32));

            torch::Tensor audioOutput = encodeAudio(audioInput);
            torch::Tensor settingsOutput = encodeSettings(settingsInput);

            torch::Tensor loss = this->loss(audioInput, settingsInput);
            lossValue = loss.item<float>();

            if (log) {
                std::cout << "Step " << step << " loss: " << lossValue
                          << std::endl;
            }

            if (!std::isfinite(lossValue)) {
                std::cerr << "ERROR: Loss is NaN/Inf at step " << step
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
