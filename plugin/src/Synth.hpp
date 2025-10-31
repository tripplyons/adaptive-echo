#pragma once

#include "Interpolation.hpp"
#include "Normalization.hpp"
#include "Oscillator.hpp"
#include "Parameters.hpp"
#include "TrainingEnvelope.hpp"
#include <iostream>
#include <vector>

using namespace std;

class Synth {
  public:
    Synth(SynthesizerParameters params) : params(params) {}
    ~Synth() {}
    const vector<double> synthesize(vector<double> time) {
        mt19937 rng(0);
        vector<double> output(time.size());
        for (unsigned int i = 0; i < time.size(); i++) {
            output[i] = double(
                osc_params(rng, time[i], params.oscillatorA.lowModulation));
            params.detach();
        }
        return output;
    }
    pair<SynthesizerParameters, double> simpleGradient(vector<double> time,
                                         vector<double> target) {
        mt19937 rng(0);
        autodiff::VectorXvar paramsVector = params.toVectorX();
        SynthesizerParameters currentParams(paramsVector);
        vector<autodiff::var> output(time.size());
        for (unsigned int i = 0; i < time.size(); i++) {
            output[i] = osc_params(rng, time[i],
                                   currentParams.oscillatorA.lowModulation);
        }
        autodiff::var loss = 0;
        for (unsigned int i = 0; i < time.size(); i++) {
            loss += (output[i] - target[i]) * (output[i] - target[i]);
        }
        autodiff::VectorXvar gradients = autodiff::gradient(loss, paramsVector);
        SynthesizerParameters newParams(gradients);
        return make_pair(newParams, double(loss));
    }
    void simpleTraining(vector<double> time, vector<double> target,
                        float learningRate, bool printLoss, unsigned int gradientBatchSize) {
        vector<double> averageGradients(params.toVector().size());
        double totalLoss = 0.0;
        for (unsigned int i = 0; i < time.size(); i += gradientBatchSize) {
            unsigned int batchEnd = i + gradientBatchSize;
            if (batchEnd > time.size()) {
                batchEnd = time.size();
            }
            vector<double> timeBatch = vector<double>(time.begin() + i, time.begin() + batchEnd);
            vector<double> targetBatch = vector<double>(target.begin() + i, target.begin() + batchEnd);
            pair<SynthesizerParameters, double> result =
                simpleGradient(timeBatch, targetBatch);
            SynthesizerParameters gradients = result.first;
            double loss = result.second;
            vector<autodiff::var> gradientsVar = gradients.toVector();
            for (unsigned int j = 0; j < gradientsVar.size(); j++) {
                averageGradients[j] += double(gradientsVar[j]);
            }
            totalLoss += loss;
        }
        if (printLoss) {
            cout << "Loss: " << totalLoss / time.size() << endl;
        }
        for (unsigned int i = 0; i < averageGradients.size(); i++) {
            averageGradients[i] /= time.size();
        }
        vector<autodiff::var> paramsVar = params.toVector();
        vector<double> newParams(averageGradients.size());
        for (unsigned int i = 0; i < averageGradients.size(); i++) {
            newParams[i] =
                double(paramsVar[i]) - learningRate * double(averageGradients[i]);
        }
        vector<autodiff::var> newParamsVar(newParams.size());
        for (unsigned int i = 0; i < newParams.size(); i++) {
            newParamsVar[i] = autodiff::var(newParams[i]);
        }
        params = SynthesizerParameters(newParamsVar);
    }
    SynthesizerParameters params;

    // Helper for synth_sample
    SingleOscillatorParameters
    single_osc_linear_interp(const SingleOscillatorParameters &a,
                             const SingleOscillatorParameters &b,
                             const autodiff::var &t) {
        SingleOscillatorParameters result;
        result.frequency = linear_interp(a.frequency, b.frequency, t);
        result.phaseShift = linear_interp(a.phaseShift, b.phaseShift, t);
        result.warmth = linear_interp(a.warmth, b.warmth, t);
        result.harshness = linear_interp(a.harshness, b.harshness, t);
        result.amplitude = linear_interp(a.amplitude, b.amplitude, t);
        result.noiseLevel = linear_interp(a.noiseLevel, b.noiseLevel, t);
        return result;
    }

    // Helper for synth_sample
    autodiff::var single_osc_uniform(std::mt19937 &rng, const var &time,
                                     const SingleOscillatorParameters &params,
                                     const var &modulation,
                                     const var &fm_amount) {
        return osc_uniform(rng, time, params.frequency, params.phaseShift,
                           params.warmth, params.harshness, params.amplitude,
                           params.noiseLevel, modulation, fm_amount);
    }

    autodiff::var synth_sample(std::mt19937 &rng, const autodiff::var &time,
                               const SynthesizerParameters &params) {
        using autodiff::var;

        // Calculate envelopes
        var env_vol_a = env_uniform(time, params.volumeA);
        var env_vol_b = env_uniform(time, params.volumeB);
        var env_mod = env_uniform(time, params.highLowModulation);

        // Interpolate settings from modulation
        SingleOscillatorParameters osc_a_settings_modulated =
            single_osc_linear_interp(params.oscillatorA.lowModulation,
                                     params.oscillatorA.highModulation,
                                     env_mod);
        SingleOscillatorParameters osc_b_settings_modulated =
            single_osc_linear_interp(params.oscillatorB.lowModulation,
                                     params.oscillatorB.highModulation,
                                     env_mod);

        // Calculate frequency modulation amount
        var env_fm = env_uniform(time, params.fmAmount);
        var fm_amount =
            linear_interp(params.startFmAmount, params.endFmAmount, env_fm);

        // Calculate oscillators
        var osc_b =
            single_osc_uniform(rng, time, osc_b_settings_modulated, 0.0, 0.0);
        var osc_a = single_osc_uniform(rng, time, osc_a_settings_modulated,
                                       osc_b, fm_amount);

        // Multiply by envelopes
        osc_a *= env_vol_a;
        osc_b *= env_vol_b;

        return osc_a + osc_b;
    }
};