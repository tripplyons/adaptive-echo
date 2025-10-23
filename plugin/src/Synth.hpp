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
            output[i] = double(osc_params(
                rng, time[i], params.oscillatorA.lowModulation));
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
            output[i] = osc_params(
                rng, time[i], currentParams.oscillatorA.lowModulation);
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