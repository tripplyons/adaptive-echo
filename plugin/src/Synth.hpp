#pragma once

#include "Interpolation.hpp"
#include "Oscillator.hpp"
#include "Parameters.hpp"

class Synth {
  public:
    Synth(SynthesizerParameters params) : params(params) {}
    ~Synth() {}
    SynthesizerParameters simpleGradient(std::vector<float> time,
                                         std::vector<float> target,
                                         bool printLoss) {
        std::mt19937 rng(0);
        autodiff::VectorXvar paramsVector = params.toVectorX();
        SynthesizerParameters currentParams(paramsVector);
        std::vector<autodiff::var> output(time.size());
        for (unsigned int i = 0; i < time.size(); i++) {
            output[i] = osc_uniform(
                rng, time[i],
                sigmoid(currentParams.oscillatorA.lowModulation.frequency),
                sigmoid(currentParams.oscillatorA.lowModulation.phaseShift),
                sigmoid(currentParams.oscillatorA.lowModulation.warmth),
                sigmoid(currentParams.oscillatorA.lowModulation.harshness),
                sigmoid(currentParams.oscillatorA.lowModulation.amplitude),
                sigmoid(currentParams.oscillatorA.lowModulation.noiseLevel),
                0.0, 0.0);
        }
        autodiff::var loss = 0;
        for (unsigned int i = 0; i < time.size(); i++) {
            loss += (output[i] - target[i]) * (output[i] - target[i]);
        }
        if (printLoss) {
            std::cout << "Loss: " << loss << std::endl;
        }
        autodiff::VectorXvar gradients = autodiff::gradient(loss, paramsVector);
        return SynthesizerParameters(gradients);
    }
    void simpleTraining(std::vector<float> time, std::vector<float> target,
                        float learningRate, bool printLoss) {
        SynthesizerParameters gradients = simpleGradient(time, target, printLoss);
        std::vector<autodiff::var> gradientsVar = gradients.toVector();
        std::vector<autodiff::var> paramsVar = params.toVector();

        std::vector<double> newParams(gradientsVar.size());
        for (unsigned int i = 0; i < gradientsVar.size(); i++) {
            newParams[i] =
                double(paramsVar[i]) - learningRate * double(gradientsVar[i]);
        }
        std::vector<autodiff::var> newParamsVar(newParams.size());
        for (unsigned int i = 0; i < newParams.size(); i++) {
            newParamsVar[i] = autodiff::var(newParams[i]);
        }
        params = SynthesizerParameters(newParamsVar);
    }
    SynthesizerParameters params;
};