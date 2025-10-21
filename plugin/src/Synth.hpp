#pragma once

#include "Interpolation.hpp"
#include "Normalization.hpp"
#include "Oscillator.hpp"
#include "Parameters.hpp"
#include <iostream>
#include <vector>

using namespace std;

class Synth {
  public:
    Synth(SynthesizerParameters params) : params(params) {}
    ~Synth() {}
    vector<double> synthesize(vector<float> time) {
        mt19937 rng(0);
        vector<double> output(time.size());
        for (unsigned int i = 0; i < time.size(); i++) {
            output[i] = double(osc_uniform(
                rng, time[i],
                sigmoid(params.oscillatorA.lowModulation.frequency),
                sigmoid(params.oscillatorA.lowModulation.phaseShift),
                sigmoid(params.oscillatorA.lowModulation.warmth),
                sigmoid(params.oscillatorA.lowModulation.harshness),
                sigmoid(params.oscillatorA.lowModulation.amplitude),
                sigmoid(params.oscillatorA.lowModulation.noiseLevel), 0.0,
                0.0));
            params.detach();
        }
        return output;
    }
    SynthesizerParameters simpleGradient(vector<float> time,
                                         vector<float> target, bool printLoss) {
        mt19937 rng(0);
        autodiff::VectorXvar paramsVector = params.toVectorX();
        SynthesizerParameters currentParams(paramsVector);
        vector<autodiff::var> output(time.size());
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
            cout << "Loss: " << loss << endl;
        }
        autodiff::VectorXvar gradients = autodiff::gradient(loss, paramsVector);
        return SynthesizerParameters(gradients);
    }
    void simpleTraining(vector<float> time, vector<float> target,
                        float learningRate, bool printLoss) {
        SynthesizerParameters gradients =
            simpleGradient(time, target, printLoss);
        vector<autodiff::var> gradientsVar = gradients.toVector();
        vector<autodiff::var> paramsVar = params.toVector();

        vector<double> newParams(gradientsVar.size());
        for (unsigned int i = 0; i < gradientsVar.size(); i++) {
            newParams[i] =
                double(paramsVar[i]) - learningRate * double(gradientsVar[i]);
        }
        vector<autodiff::var> newParamsVar(newParams.size());
        for (unsigned int i = 0; i < newParams.size(); i++) {
            newParamsVar[i] = autodiff::var(newParams[i]);
        }
        params = SynthesizerParameters(newParamsVar);
    }
    SynthesizerParameters params;
};